# Sign Language Recognition

## Project Description
This project implements a real-time sign language recognition system. It utilizes computer vision techniques to interpret hand gestures and translate them into text or other forms of communication. The system is designed to be user-friendly, providing a live video stream for gesture input and immediate recognition feedback.

## Setup Steps

### Step 1: Clone the Repository
Open your terminal or command prompt, then run:
```bash
git clone <repository_url>
cd Sign-Language-Recognition
```

### Step 2: Set Up a Virtual Environment
Run the following commands to create and activate a virtual environment:
```bash
virtualenv env
```
For Windows, run the following in your terminal to allow script execution:
```bash
Set-ExecutionPolicy Unrestricted -Scope Process
```
Then run the following to activate the environment:
```bash
.\env\Scripts\activate
```
For macOS/Linux:
```bash
source env/bin/activate
```

### Step 3: Install Required Packages
Install dependencies listed in `requirements.txt` using pip (you only do this once; next time you want to run the project, you can skip this):
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation (Optional)
Check that all dependencies are installed correctly:
```bash
pip list
```

### Step 5: Start the Application
Run the following command to start the application:
```bash
python web_app.py
```

### Step 6: Use the Application
A video stream will open, allowing real-time sign language recognition. Ensure your camera is connected and gestures are made within its view.

## Usage
(Further details on how to use the application, e.g., specific gestures, UI elements, etc., can be added here.)

## Contributing
Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
