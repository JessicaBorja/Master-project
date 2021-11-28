import hydra
import os
import json


# Merge datasets using json files
def merge_datasets(directory_list, output_dir):
    new_data = {"train": {}, "validation": {}}
    for dir in directory_list:
        json_path = os.path.join(dir, "episodes_split.json")
        with open(json_path) as f:
            data = json.load(f)

        # Rename episode numbers if repeated
        data_split = list(data.keys())
        episode = 0
        for split in data_split:
            dataset_name = os.path.basename(os.path.normpath(dir))
            for key in data[split].keys():
                new_data[split]["/%s/%s" % (dataset_name, key)] = \
                    data[split][key]
                episode += 1
    # Write output
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    with open(output_dir + '/episodes_split.json', 'w') as outfile:
        json.dump(new_data, outfile, indent=2)


@hydra.main(config_path="../config", config_name="cfg_merge_dataset")
def main(cfg):
    merge_datasets(data_lst, cfg.output_dir)


if __name__ == "__main__":
    main()