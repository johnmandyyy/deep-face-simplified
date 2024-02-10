from helpers import Validators, Operations
import pandas as pd
from deepface import DeepFace

class FaceSearch:

    def __init__(self):
        """
        Creates a file.
        """

        # Initialize instances.
        self.validator = Validators()
        self.operations = Operations()

        # Initialize targeted directories.
        self.operations.initialize_directories("face_categories")

    def buildPickle(self, db_path: str, model_name: str) -> bool:
        
        """
        Can be used to build a 
        pickle file from the given dataset, apply model
        
        Params:
            db_path: -> Location of folder
            model_name: -> Models:
                - "VGG-Face"
                - "OpenFace"
                - "Facenet"
                - "Facenet512"
                - "DeepFace"
                - "DeepID"
                - "Dlib"
                - "ArcFace"
                - "SFace"
                - "Emotion"
                - "Age"
                - "Gender"
                - "Race"

        Returns:
            True if the model was 
            successfully built, otherwise False
        """

        # Usage
        """
            instance = FaceSearch()
            instance.buildPickle("face_categories/black", "ArcFace")
        """
        return self.operations.createModel(db_path, model_name)


    def classifyRecords(self, db_path: str, category: list) -> pd.DataFrame:
        
        """
        Clasify all records based on specific category,
        Moves images to the designated directory,
        returns a pandas dataframe from the given set.

        Parameters:
            db_path (str): Path to the database.
            category (list): List of categories.

        Raises:
            TypeError: If db_path is not a string or category is not a list.
            IndexError: If the length of category is less than 1.
            ValueError: If any category is not in the accepted_params list.

        """

        # Usage
        """
            instance = FaceSearch()
            data_frame = instance.classifyRecords("face_db", ["race"])
            data_frame.to_csv("output_file.csv", index=False)

        """
        # Validates argument if correct.
        self.validator.validate_record(db_path, category)
        return self.operations.categorize_persons(db_path)


