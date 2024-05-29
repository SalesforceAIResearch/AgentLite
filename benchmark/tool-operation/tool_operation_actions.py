from agentlite.actions.BaseAction import BaseAction
from agentlite.actions import FinishAct 

class finish(BaseAction):
    def __init__(self, env=None) -> None:
        action_name = FinishAct.action_name # substitute with the BaseAgent finish action with this new action
        action_desc = "Return an answer and finish the task"
        params_doc = {
            "response": "this is the finish action response. As simple as possible."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)

    def __call__(self, response):
        params = {"answer": response}
        action = ('finish', params)
        observation, reward, done, _ = self.env.step(action)
        return observation

    
# Define Todo Actions
class get_user_current_date(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_user_current_date"
        action_desc = "Get the current date of the user"
        params_doc = {
            "None": "No input required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_user_current_location(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_user_current_location"
        action_desc = "Get the current location of the user"
        params_doc = {
            "None": "No input required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_projects(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_projects"
        action_desc = "Get the list of projects"
        params_doc = {
            "None": "No input required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class update_project(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "update_project"
        action_desc = "Update the project details"
        params_doc = {
            "project_id": "(Type: string): The ID of the project to update",
            "is_favorite": "(Type: boolean): The favorite status of the project"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_tasks(BaseAction): 
    def __init__(self, env = None) -> None:
        action_name = "get_tasks"
        action_desc = "Get the list of tasks for a project"
        params_doc = {
            "project_id": "(Type: string): The ID of the project to get tasks for"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_task_description(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_task_description"
        action_desc = "Get the description of a task"
        params_doc = {
            "task_id": "(Type: string): The ID of the task to get description for"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_task_duration(BaseAction): 
    def __init__(self, env = None) -> None:
        action_name = "get_task_duration"
        action_desc = "Get the duration of a task"
        params_doc = {
            "task_id": "(Type: string): The ID of the task to get duration for"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class complete_task(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "complete_task"
        action_desc = "Mark a task as completed"
        params_doc = {
            "task_id": "(Type: string): The ID of the task to mark as completed"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class update_task(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "update_task"
        action_desc = "Update the due date of a task"
        params_doc = {
            "task_id": "(Type: string): The ID of the task to update",
            "due_date": "(Type: string): The new due date for the task (YYYY-MM-DD)"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class delete_task(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "delete_task"
        action_desc = "Delete a task"
        params_doc = {
            "task_id": "(Type: string): The ID of the task to delete"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
# Define Sheet Action
class open_sheet(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "open_sheet"
        action_desc = "Open a sheet"
        params_doc = {
            "name": "(Type: string): The name of the sheet to open"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class del_sheet(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "del_sheet"
        action_desc = "Delete a sheet"
        params_doc = {
            "name": "(Type: string): The name of the sheet to delete"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class freeze_data(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "freeze_data"
        action_desc = "Freeze data in a sheet"
        params_doc = {
            "dimension": "(Type: string): The dimension to freeze data in (row or column)",
            "num": "(Type: number): The number of rows or columns to freeze"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_A1_annotation(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_A1_annotation"
        action_desc = "Get the annotation at A1"
        params_doc = {
            "row": "(Type: number): The row of the annotation",
            "col": "(Type: number): The column of the annotation"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class insert_cols(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "insert_cols"
        action_desc = "Insert columns into a sheet"
        params_doc = {
            "values_list": "(Type: list): The list of values to insert",
            "col_idx": "(Type: number): The index of the column to insert the values into"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class insert_rows(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "insert_rows"
        action_desc = "Insert rows into a sheet"
        params_doc = {
            "values_list": "(Type: list): The list of values to insert",
            "row_idx": "(Type: number): The index of the row to insert the values into"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class delete_batch_data(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "delete_batch_data"
        action_desc = "Delete batch data from a sheet"
        params_doc = {
            "dimension": "(Type: string): The dimension to delete data from (row or column)",
            "index_list": "(Type: list): The list of indices to delete data from"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class update_cell(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "update_cell"
        action_desc = "Update a cell in a sheet"
        params_doc = {
            "position": "(Type: string): The position of the cell to update",
            "value": "(Type: string): The value to update the cell with"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class update_cell_by_formula(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "update_cell_by_formula"
        action_desc = "Update a cell by formula in a sheet"
        params_doc = {
            "start_position": "(Type: string): The start position of the range to update",
            "end_position": "(Type: string): The end position of the range to update",
            "position_list": "(Type: list): The list of positions to update by formula",
            "operator": "(Type: string): The operator to use in the formula",
            "result_position": "(Type: string): The position to store the result of the formula"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class update_range(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "update_range"
        action_desc = "Update a range in a sheet"
        params_doc = {
            "start_position": "(Type: string): The start position of the range to update",
            "end_position": "(Type: string): The end position of the range to update",
            "values_list": "(Type: list): The list of values to update the range with"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class sort_sheet_by_col(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "sort_sheet_by_col"
        action_desc = "Sort a sheet by column"
        params_doc = {
            "col_num": "(Type: number): The number of the column to sort by",
            "order": "(Type: string): The order to sort by (ascending or descending)"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class merge_cells(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "merge_cells"
        action_desc = "Merge cells in a sheet"
        params_doc = {
            "start_position": "(Type: string): The start position of the range to merge",
            "end_position": "(Type: string): The end position of the range to merge"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class update_note(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "update_note"
        action_desc = "Update a note in a sheet"
        params_doc = {
            "position": "(Type: string): The position of the cell to update the note for",
            "content": "(Type: string): The content of the note to update"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_all_values(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_all_values"
        action_desc = "Get all values in a sheet"
        params_doc = {
            "None": "No input required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_range_values(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_range_values"
        action_desc = "Get values in a range in a sheet"
        params_doc = {
            "start_position": "(Type: string): The start position of the range to get values from",
            "end_position": "(Type: string): The end position of the range to get values from"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_cell_value(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_cell_value"
        action_desc = "Get the value of a cell in a sheet"
        params_doc = {
            "position": "(Type: string): The position of the cell to get the value for"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_value_by_formula(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_value_by_formula"
        action_desc = "Get the value of a cell by formula in a sheet"
        params_doc = {
            "start_position": "(Type: string): The start position of the range to get values by formula",
            "end_position": "(Type: string): The end position of the range to get values by formula",
            "position_list": "(Type: list): The list of positions to get values by formula",
            "operator": "(Type: string): The operator to use in the formula"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class filter_cells(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "filter_cells"
        action_desc = "Filter cells in a sheet"
        params_doc = {
            "query": "(Type: string): The query to filter cells by",
            "in_row": "(Type: boolean): Whether to filter cells in the row",
            "in_column": "(Type: boolean): Whether to filter cells in the column"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_note(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_note"
        action_desc = "Get the note of a cell in a sheet"
        params_doc = {
            "position": "(Type: string): The position of the cell to get the note for"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation


def get_todo_actions(env):
    todo_actions = [
        get_user_current_date(env),
        get_user_current_location(env),
        get_projects(env),
        update_project(env),
        get_tasks(env),
        get_task_description(env),
        get_task_duration(env),
        complete_task(env),
        update_task(env),
        delete_task(env),
        finish(env)
    ]
    return todo_actions

def get_sheet_actions(env):
    sheet_actions = [
        open_sheet(env),
        del_sheet(env),
        freeze_data(env),
        get_A1_annotation(env),
        insert_cols(env),
        insert_rows(env),
        delete_batch_data(env),
        update_cell(env),
        update_cell_by_formula(env),
        update_range(env),
        sort_sheet_by_col(env),
        merge_cells(env),
        update_note(env),
        get_all_values(env),
        get_range_values(env),
        get_cell_value(env),
        get_value_by_formula(env),
        filter_cells(env),
        get_note(env),
        finish(env)
    ]
    return sheet_actions