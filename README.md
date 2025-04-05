# hAIt (Have it, AI, Get It)

hAIt is an application designed to extract exercises from exam models, solve them using AI, and export the results in a PDF format. The app leverages AI to provide accurate and efficient solutions, making it a powerful tool for students.

## Features
- Extract exercises from exam models.
- Solve exercises using AI.
- Export results in a clean and organized PDF format.

## Installation
```bash
pip install -r requirements.txt
```

```bash
1. # Ensure your models are placed in the 'models/' folder
2. # Insert AI api key on env.py
```
## Usage
Run the application with the following command:
```bash
python main.py
```


## AI Integration
The app uses [**TogetherAI**](https://api.together.ai/) for its high free-tier capabilities but can also work seamlessly with **ChatGPT** for solving exercises.

## Output
The results of the solved exercises are exported as a PDF, providing a convenient and shareable format.