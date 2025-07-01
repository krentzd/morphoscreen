import xmltodict
import os
from glob import glob
import tifffile
from tqdm import tqdm
import imageio
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import pandas as pd
import math
import argparse

def is_float(string):
    if string.replace(".", "").isnumeric():
        return True
    else:
        return False

def parse_string(input_string):
    input_string = input_string.replace(' ', '')
    output_string_list = []
    output_string_list.append(input_string[0] + "'")
    for i in range(1,len(input_string)):
        if input_string[i] == "{":
            output_string_list.append("{'")
        elif input_string[i] == "}" and input_string[i-1] != "}":
            output_string_list.append("'}")
        elif input_string[i] == ":":
            if input_string[i+1] != "{" and input_string != "[":
                output_string_list.append("':'")
            else:
                output_string_list.append("':")
        elif input_string[i] == ",":
            if input_string[i-1] == '}':
                output_string_list.append(",'")
            else:
                output_string_list.append("','")
        else:
            output_string_list.append(input_string[i])

    output_string = ''.join(output_string_list)

    final_string_list = []
    for sub_string in output_string.split("'"):
        if is_float(sub_string.replace("[", '').replace("]", '').replace("-", '').replace("+", '')):
            final_string_list.append(sub_string)
        elif any(s in ["{", "}", ":", ","] for s in sub_string):
            final_string_list.append(sub_string)
        else:
            final_string_list.append("'" + sub_string + "'")

    final_string = ''.join(final_string_list)

    return final_string

class OperaPhenixDataset():
    def __init__(self, root, plate_map, channels='all', transform=None):

        self.file_dir = root
        self.plate_map = plate_map
        self.transform = transform
        self.channel_name_dict = {'HOECHST33342': 'Hoechst',
                                  'Alexa488Restrictedemission': 'SytoxGreen',
                                  'Alexa568': 'FM4-64',
                                  'FM4-64': 'FM4-64',
                                  'Brightfield': 'Brightfield'}

        self.channel_dict = dict()

        xml_path = os.path.join(self.file_dir, 'Index.xml')
        with open(xml_path, 'r', encoding='utf-8') as file:
            xml_file = file.read()

        self.xml_dict = xmltodict.parse(xml_file)

        self.make_channel_dict()
        self.make_image_path_dict()
        self.make_dataset_indices()

        if channels == 'all':
            self._channels = [ch for ch in ['Hoechst', 'SytoxGreen', 'FM4-64', 'Brightfield'] if ch in self.channel_dict.keys()]
        else:
            self._channels = channels

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes())}

    def make_channel_dict(self):
        for channel_num in range(self.num_channels()):
            # Dirty, but works for now
            input_str = self.xml_dict['EvaluationInputData']['Maps']['Map'][0]['Entry'][channel_num]['FlatfieldProfile'].replace('Acapella:2013', 'Acapella_2013')

            parsed_string = parse_string(input_str)
            parsed_dict = eval(parsed_string)

            self.channel_dict[self.channel_name_dict[parsed_dict['ChannelName']]] = channel_num + 1

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channel_list):
        self._channels = [ch for ch in channel_list if ch in self.channel_dict.keys()]

    def make_image_path_dict(self):
        xml_im_dict = self.xml_dict['EvaluationInputData']['Images']['Image']
        im_list = [(xml_im_dict[i]['Row'], xml_im_dict[i]['Col'], xml_im_dict[i]['FieldID'], xml_im_dict[i]['ChannelID'], xml_im_dict[i]['URL']) for i in range(len(self.xml_dict['EvaluationInputData']['Images']['Image']))]
        self.path_dict = dict(zip([(int(item[0]), int(item[1]), int(item[2]), int(item[3])) for item in im_list], [item[4] for item in im_list]))

    def num_channels(self):
        return len(self.xml_dict['EvaluationInputData']['Maps']['Map'][0]['Entry'])

    def num_rows(self):
        xml_well_dict = self.xml_dict['EvaluationInputData']['Wells']['Well']
        rows = [int(xml_well_dict[i]['Row']) for i in range(len(xml_well_dict))]
        return max(rows)

    def num_columns(self):
        xml_well_dict =self.xml_dict['EvaluationInputData']['Wells']['Well']
        columns = [int(xml_well_dict[i]['Col']) for i in range(len(xml_well_dict))]
        return max(columns)

    def num_fields(self):
        fields = [int(self.xml_dict['EvaluationInputData']['Images']['Image'][i]['FieldID']) for i in range(self.__len__())]
        return max(fields)

    def classes(self):
        return sorted(list(set([item[0] for item in self.plate_map.values()])))

    def __len__(self):
        return math.floor(len(self.xml_dict['EvaluationInputData']['Images']['Image']) / self.num_channels())

    def read_image_stack(self, row, col, field, channels, dtype='uint8', clip=True, **kwargs):
        im_path_list = []
        for channel in channels:
            ch_num = self.channel_dict[channel]
            im_base_path = self.path_dict[(row, col, field, ch_num)]
            im_path = os.path.join(self.file_dir, im_base_path)
            im_path_list.append(im_path)

        im_list = []
        for im_path in  im_path_list:
            im = tifffile.imread(im_path)

            if clip:
                p_bot, p_top = np.percentile(im, kwargs.get('clip_lower', 0.1)), np.percentile(im, kwargs.get('clip_upper', 99.9))
                im = np.clip(im, p_bot, p_top)

            im = exposure.rescale_intensity(im, out_range=dtype)

            im_list.append(im)

        if kwargs.get('return_path', False):
            return np.stack(im_list), im_path_list
        else:
            return np.stack(im_list)

    def read_image_from_name(self, name, state, field, channels=None):
        inv_plate_map = {value: key for key, value in self.plate_map.items()}

        row, col = inv_plate_map[(name, state)]

        if channels == None:
            channels = self.channels

        return self.read_image_stack(row, col, field, channels)

    def make_dataset_indices(self):
        self.idx_list = []
        for row_idx, col_idx in list(self.plate_map.keys()):
                for field_idx in range(1, self.num_fields() + 1):
                    self.idx_list.append((row_idx, col_idx, field_idx))

    def __getitem__(self, idx):
        row, col, field = self.idx_list[idx]

        im = self.read_image_stack(row, col, field, self.channels)
        label = self.plate_map[(row, col)][0]

        if self.transform:
            im = self.transform(im)

        return im, self.class_to_idx[label]

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def get_plate_map(path):

    row_dict = dict(zip([chr(i) for i in range(ord('A'), ord('P')+1)], [i for i in range(1,17)]))

    plate_map_df = pd.read_csv(plate_map_path, delimiter=',', usecols=['cond', 'Destination well', 'rep']).dropna()
    plate_map_df['Row'] = plate_map_df['Destination well'].map(lambda x: x[:1]).map(row_dict)
    plate_map_df['Column'] = plate_map_df['Destination well'].map(lambda x: int(x[1:]))
    plate_map_df['cond'] = plate_map_df['cond'].replace({'controlDMSO': 'DMSO', 'controlwater': 'Water'})
    plate_map_df['rep'] = plate_map_df['rep']

    plate_map_keys = [(row, col) for (row, col) in zip(list(plate_map_df.Row.values), list(plate_map_df.Column.values))]
    plate_map_values = list(zip(plate_map_df.cond.values, plate_map_df.rep.values))
    plate_map = dict(zip(plate_map_keys, plate_map_values))

    return plate_map

cglu_path_dict = {('P1', 'conf', 96): '240628 2 replicates/240628 plate 1/2024-06-28 nm jp confo 63x plate1__2024-06-28T13_39_48-Measurement 1',
                  ('P2', 'conf', 96): '240628 2 replicates/240701 plate 2/2024-07-01 nm jp pl2806 confo 63x plate2__2024-07-01T10_34_41-Measurement 1',
                  ('P3', 'conf', 96): '240709 2 replicates/2024-07-09 nm jp pl0507 confo 63x plate3__2024-07-09T10_30_16-Measurement 1',
                  ('P4', 'conf', 96): '240709 2 replicates/2024-07-09 nm jp pl0507 confo 63x plate4__2024-07-09T14_04_30-Measurement 2',
                  ('P6', 'conf', 96): '240724 1 plate plate6/2024-07-25 nm jp pl1807 confo 63x pl6__2024-07-25T07_58_07-Measurement 1',
                  ('P7', 'conf', 96): '241031 plate 7 other handling/2024-10-31 nm jp confo 63x pl7__2024-10-31T13_12_52-Measurement 1',
                  ('P8', 'conf', 96): '241104 plate 8 other handling/2024-11-04 nm jp confo 63x pl8__2024-11-04T10_48_44-Measurement 1',
                  ### Non confocal
                  ('P1', 'non_conf', 96): '240628 2 replicates/240628 plate 1/2024-06-28 nm jp no confo 63x__2024-06-28T15_42_05-Measurement 1',
                  ('P2', 'non_conf', 96): '240628 2 replicates/240701 plate 2/2024-07-01 nm jp pl2806 no confo 63x pl2__2024-07-01T08_51_34-Measurement 1',
                  ('P3', 'non_conf', 96): '240709 2 replicates/2024-07-09 nm jp pl0507 no confo 63x plate 3__2024-07-09T08_52_39-Measurement 1',
                  ('P4', 'non_conf', 96): '240709 2 replicates/2024-07-09 nm jp pl0507 no confo 63x plate 4__2024-07-09T12_32_01-Measurement 1',
                  ('P6', 'non_conf', 96): '240724 1 plate plate6/2024-07-24 nm jp pl1807 no confo 63x pl6__2024-07-24T08_04_17-Measurement 1',
                  ('P7', 'non_conf', 96): '241031 plate 7 other handling/2024-10-31 nm jp no confo 63x pl7__2024-10-31T11_43_37-Measurement 1',
                  ('P8', 'non_conf', 96): '241104 plate 8 other handling/2024-11-04 nm jp no confo 63x pl8__2024-11-04T09_11_34-Measurement 1',
                  ('P9', 'non_conf', 96): '250114 plate 9/2025-01-14 nm jp no confo pl9__2025-01-14T09_38_09-Measurement 1',
                  ('P10', 'non_conf', 96): '250128 plate10/2025-01-28 nm bym no confo 63x pl10__2025-01-28T09_45_51-Measurement 1',

                  ('P1', 'non_conf', 384): '250320 1rst trial 384w plate/2025-03-20 nm bym no confo 63x pl384_1 2__2025-03-20T13_47_46-Measurement 1'
                  }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plate', required=True)
    parser.add_argument('--plate_type', default=96)
    parser.add_argument('--by_mic', action='store_true', default=False)
    parser.add_argument('--confocal', action='store_true', default=False)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--save_as_tiff', action='store_true', required=True)

    args = parser.parse_args()

    im_mode = 'conf' if args.confocal else 'non_conf'

    plate_map_path = f'../plate-layouts/Cglu-plate-maps/{args.plate_type}w_22atb_plate{args.plate}_DK.csv'
    plate_map = get_plate_map(plate_map_path)

    dir_path = os.path.join('../../Datasets/', cglu_path_dict[(f'P{args.plate}', im_mode, args.plate_type)])
    data_path = os.path.join(dir_path, 'Images')

    op_data = OperaPhenixDataset(data_path, plate_map)

    if args.by_mic:
        target_dir = f'../../Preprocessed_datasets/{args.save_dir}/MIC_ordering/Plate_by_MIC_{args.plate}'
    else:
        target_dir = f'../../Preprocessed_datasets/{args.save_dir}/with_clip/Plate_{args.plate}'

    make_dir(target_dir)

    if args.confocal:
        channels = ['Hoechst', 'Hoechst', 'FM4-64']
    else:
        channels = op_data.channels

    for idx in tqdm(op_data.idx_list):

        try:
            condition_name, rep_num = op_data.plate_map[(idx[0],idx[1])]
            if args.by_mic:
                cmpd_name, mic_name = condition_name.split('_')
                new_dir = os.path.join(target_dir, f'Rep_{rep_num}', f'{mic_name}', f'{cmpd_name}')
            else:
                new_dir = os.path.join(target_dir, f'Rep_{rep_num}', f'{condition_name}')

            make_dir(new_dir)

            im_as_tiff_stack, im_paths = op_data.read_image_stack(idx[0], idx[1], idx[2], channels, clip=True, return_path=True)

            im_as_tiff_stack = np.moveaxis(np.array(im_as_tiff_stack), 0, -1)

            im_base_path = os.path.basename(im_paths[0]).replace('ch1', 'ch123').replace('tiff', 'png')

            save_path = os.path.join(new_dir, im_base_path)
            if args.save_as_tiff:
                tifffile.imwrite(save_path , im_as_tiff_stack)
            else:
                imageio.imwrite(save_path , np.squeeze(im_as_tiff_stack))
        except KeyError:
            print(f'KeyError: Could not find {idx}!')
        except ValueError:
            print(f'ValueError: {idx}')
