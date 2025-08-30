Setup Steps
Step 1: Clone the Repository
Open your terminal or command prompt, then run:

Step 2: Set Up a Virtual Environment
	Run the following commands to create and activate a virtual environment:
	virtualenv env

	run the following in your terminal:
	Set-ExecutionPolicy Unrestricted -Scope Process

	then run the following to activate the environment:
	.\env\Scripts\activate

Step 3: Install Required Packages
	Install dependencies listed in requirements.txt using pip (you only do this once, next time you want to run the project, you skip this)

Step 4: Verify Installation
	(OPTIONAL, NOT NECESSARY) Check that all dependencies are installed correctly:
	pip list

Step 5: Start the application by running:
	python app.py

Step 6: Use the Application
	A video stream will open, allowing real-time sign language recognition.
	Ensure your camera is connected and gestures are made within its view.
