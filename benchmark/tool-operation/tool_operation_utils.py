import json
import os

class ToolDataset:
    def __init__(self, test_file) -> None:
        super().__init__()
        self._load_data(test_file)

    def _load_data(self, test_file_path):
        
        data = None
        with open(test_file_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        self.goals = [ i["goal"] for i in data ] 
        
        if "answer" in data[0]["additional_info"]:
            self.ground_truths = [ i["additional_info"]["answer"] for i in data ]

        if "subgoals" in data[0]:
            self.ground_truth_subgoals = [ i["subgoals"] for i in data ]

        if "init_config" in data[0]["additional_info"]: 
            self.init_configs = [ i["additional_info"]["init_config"] for i in data ]
        
        if "goal_type" in data[0]["additional_info"]: 
            self.goal_types = [ i["additional_info"]["goal_type"] for i in data ]

        if "tool" in data[0]["additional_info"]: 
            self.tools = [ i["additional_info"]["tool"] for i in data ]
        
        if "difficulty" in data[0]:
            self.difficulties = [ i["difficulty"] for i in data ]
        
    def __len__(self):
        return len(self.goals)

def load_dataset(dataset_name, dataset_dir="data"):
    dataset_file = os.path.join(dataset_dir, dataset_name, "test.jsonl")
    if dataset_name == "tool-query" or dataset_name == "tool-operation":
        dataset = ToolDataset(test_file=dataset_file)
    else:
        raise NotImplementedError("Dataset {} not implemented".format(dataset_name))
    return dataset

def get_data(idx, data_name):
    assert data_name in ["tool-query", "tool-operation"]
    dataset = load_dataset(data_name)
    dataset_i = dict()
    dataset_i["goal"] = dataset.goals[idx]
    dataset_i["ground_truth"] = dataset.ground_truths[idx]
    dataset_i["ground_truth_subgoals"] = dataset.ground_truth_subgoals[idx]
    dataset_i["tool"] = dataset.tools[idx]
    if hasattr(dataset, "action_path"):
        # sheet task does not have action_path
        dataset_i["ground_truth_action_path"] = dataset.action_path[idx]
    if dataset_i["tool"] == "weather":
        #! Same operation for todo
        dataset_i["current_date"] = dataset.init_configs[idx]["current_date"]
        dataset_i["current_location"] = dataset.init_configs[idx]["current_location"]
    return dataset_i