import os
import pickle

class StandardAgent():

    def __init__(self, experiment_dir):

        self.experiment_dir = "saved_models/" + experiment_dir + "/"
        self.model_location = self.experiment_dir + "model.h5"
        self.dict_location =  self.experiment_dir + "status.p"

        self.scores = []
        self.total_t = 0

        os.makedirs(self.experiment_dir, exist_ok=True)

    def load_state_from_dict(self):

        if os.path.exists(self.dict_location):
            with open(self.dict_location, 'rb') as md:
                model_dict = pickle.load(md)
        else:
            print("No model dict exists yet!")
            model_dict = {}

        # Initialise standard state if empty, else flexible
        
        for k, v in model_dict.items():
            if k in ("scores", "total_t"):
                continue 
            else:
                setattr(self, k, v)

        return model_dict

    def return_state_dict(self):
        """Open the model dict to view what models we have."""
        
        if os.path.exists(self.model_dict_file):
            with open(self.model_dict_file, 'rb') as md:
                model_dict = pickle.load(md)
        else:
            print("Model dict file does not exist for viewing, yet")
            model_dict = None

        return model_dict

    def load_state(self):
        raise NotImplementedError(
            "To be implemented by the inheriting agent.")