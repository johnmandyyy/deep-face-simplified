import shutil
import os
import pandas as pd
from deepface import DeepFace

# built-in dependencies
import os
import pandas as pd
from tqdm import tqdm
from deepface.modules import representation, detection

from deepface.DeepFace import recognition
from deepface.modules import modeling
from deepface.models.FacialRecognition import FacialRecognition
import pickle


class Validators:

    def list_images(self, path: str) -> list:
        """
        List images in a given path
        Args:
            path (str): path's location
        Returns:
            images (list): list of exact image paths
        """
        images = []
        for r, _, f in os.walk(path):
            for file in f:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    exact_path = f"{r}/{file}"
                    images.append(exact_path)
        return images

    def validate_record(self, db_path: str, category: list) -> None:
        """
        Validates record from the given set,
        used for classifying record.

        returns -> None

        """

        # Returns type error if image directory is not a string.
        if not isinstance(db_path, str):
            raise TypeError("db_path must be an instance of string")

        # Returns type error if categories is not a list.
        if not isinstance(category, list):
            raise TypeError("category must be a list.")

        elif isinstance(category, list):

            accepted_params = ["emotion", "age", "gender", "race"]

            # Returns an index error there are no given categories.
            if len(category) < 1:
                raise IndexError("The length of category must be greater than 1.")

            for each_category in category:

                if each_category not in accepted_params:

                    # Returns type error if category given is not in the accepted parameters.
                    raise TypeError(
                        "Category entered is not available from the given {}".format(
                            accepted_params
                        )
                    )


class Operations:

    def __init__(self):
        self.validator = Validators()

    def createModel(self, db_path: str, model_name: str):

        """
        Returns:
            True if successfully built, otherwise False.
        """

        models_list = [
            "VGG-Face",
            "OpenFace",
            "Facenet",
            "Facenet512",
            "DeepFace",
            "DeepID",
            "Dlib",
            "ArcFace",
            "SFace",
            "Emotion",
            "Age",
            "Gender",
            "Race"
        ]

        if model_name not in models_list:
            raise ValueError("Given model is not in the list {}".format(models_list))

        persons = self.validator.list_images(path = db_path)
        if len(persons) == 0:

            raise ValueError(
                f"There is no image in {db_path} folder!"
                "Validate .jpg, .jpeg or .png files exist in this path.",
            )

        model: FacialRecognition = modeling.build_model(model_name)
        target_size = model.input_shape

        file_name = f"representations_{model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()
        datastore_path = f"{db_path}/{file_name}"

        representations = self.find_bulk_embeddings(
            employees=persons,
            model_name=model_name,
            target_size=target_size,
            detector_backend="retinaface",
            enforce_detection=False,
        )

        try:
            with open(datastore_path, "wb") as f:
                pickle.dump(representations, f)
                return True
        except Exception as e:
            print(e)
            return False

    def find_bulk_embeddings(
        self,
        employees: list,
        model_name: str = "VGG-Face",
        target_size: tuple = (224, 224),
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        silent: bool = False,
    ):
        """
        Find embeddings of a list of images

        Args:
            employees (list): list of exact image paths

            model_name (str): facial recognition model name

            target_size (tuple): expected input shape of facial recognition model

            detector_backend (str): face detector model name

            enforce_detection (bool): set this to False if you
                want to proceed when you cannot detect any face

            align (bool): enable or disable alignment of image
                before feeding to facial recognition model

            expand_percentage (int): expand detected facial area with a
                percentage (default is 0).

            normalization (bool): normalization technique

            silent (bool): enable or disable informative logging
        Returns:
            representations (list): pivot list of embeddings with
                image name and detected face area's coordinates
        """
        representations = []
        for employee in tqdm(
            employees,
            desc="Finding representations",
            disable=silent,
        ):
            img_objs = detection.extract_faces(
                img_path=employee,
                target_size=target_size,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
            )

            for img_obj in img_objs:
                img_content = img_obj["face"]
                img_region = img_obj["facial_area"]
                embedding_obj = representation.represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(employee)
                instance.append(img_representation)
                instance.append(img_region["x"])
                instance.append(img_region["y"])
                instance.append(img_region["w"])
                instance.append(img_region["h"])
                representations.append(instance)

        return representations

    def categorize_persons(self, db_path: str) -> None:
        """
        Categorizes person
        based on its race.

        Returns:
            instance:
                pandas.DataFrame()
        """
        # File types accepted.
        accepted_file_type = [".jpg", ".jpeg", ".png"]

        # Ensure the folder path is valid
        if os.path.exists(db_path) and os.path.isdir(db_path):

            files = os.listdir(db_path)

            persons = []

            for file_name in files:

                if any(
                    file_name.lower().endswith(file_type)
                    for file_type in accepted_file_type
                ):

                    file_path = os.path.join(db_path, file_name)
                    print("Analyzing", file_path)

                    result = DeepFace.analyze(
                        img_path=file_path, actions=["race"], enforce_detection=False
                    )

                    copy_is_success = self.copy_to_categories(file_path, result)
                    if copy_is_success == True:

                        persons.append(
                            {
                                "fileName": file_name,
                                "result": result,
                                "directory": "",
                            }
                        )

                        print(f"File {file_name} is copied.")

            else:  # End iteration.
                return pd.DataFrame(
                    persons, columns=["fileName", "result", "directory"]
                )

        else:
            raise LookupError("Directory is not found.")

    def initialize_directories(self, folder_path: str) -> None:
        """
        Initializes directories for categories.

        Parameters:
            folder_path (str): Path to the base folder.

        """

        # Check if the folder exists
        if not os.path.exists(folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been created.")
        else:

            print(f"The base folder '{folder_path}' already exists. Skipping creation.")

            # Create asian folder.
            if not os.path.exists(folder_path + "/asian"):
                # Create the folder if it doesn't exist
                os.makedirs(folder_path + "/asian")
                print("The folder for asian has been created.")

            # Create indian folder.
            if not os.path.exists(folder_path + "/indian"):
                # Create the folder if it doesn't exist
                os.makedirs(folder_path + "/indian")
                print("The folder for indian has been created.")

            # Create black folder.
            if not os.path.exists(folder_path + "/black"):
                # Create the folder if it doesn't exist
                os.makedirs(folder_path + "/black")
                print("The folder for black has been created.")

            # Create white folder.
            if not os.path.exists(folder_path + "/white"):
                # Create the folder if it doesn't exist
                os.makedirs(folder_path + "/white")
                print("The folder for white has been created.")

            # Create middle_eastern folder.
            if not os.path.exists(folder_path + "/middle_eastern"):
                # Create the folder if it doesn't exist
                os.makedirs(folder_path + "/middle_eastern")
                print("The folder for middle eastern has been created.")

            # Create latino_hispanic folder.
            if not os.path.exists(folder_path + "/latino_hispanic"):
                # Create the folder if it doesn't exist
                os.makedirs(folder_path + "/latino_hispanic")
                print("The folder for laitno hispanic has been created.")

    def copy_to_categories(self, base_dir: str, result: dict) -> bool:
        """
        Returns True if the operation
        succeeded, otherwise False.

        Parameters:
            base_dir (str): The source file path.
            result (dict): The dictionary containing result information.

        Returns:
            bool: True if the operation succeeded, otherwise False.
        """
        try:
            # Extract the dominant race from the result
            dominant_race = result[0]["dominant_race"]

            # Adjust the dominant race for underscore determiner
            if dominant_race == "latino hispanic":
                dominant_race = "latino_hispanic"
            elif dominant_race == "middle eastern":
                dominant_race = "middle_eastern"

            # Create the target directory path
            target_directory = os.path.join("face_categories", dominant_race)

            # Check if the target directory exists, create if not
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            # Copy the file to the target directory
            shutil.copy(base_dir, target_directory)

            return True
        except Exception as e:
            print(e)
            return False
