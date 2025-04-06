import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
import re
from imagehash import phash
from together import Together
from env import TOGETHER_API_KEY

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_exercises(image_path):
    """Extract individual exercises from an exam image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return [], []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply some image processing to improve OCR
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Use OCR to detect numbers followed by parentheses
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    exercise_positions = []
    for i in range(len(text_data['text'])):
        text = text_data['text'][i].strip()
        if re.match(r'^\d+\)$', text) or re.match(r'^\d+\.$', text):
            exercise_num = int(re.search(r'\d+', text).group())
            y_pos = text_data['top'][i]
            exercise_positions.append((exercise_num, y_pos))
    
    # Sort by y-position
    exercise_positions.sort(key=lambda x: x[1])
    
    exercises = []
    exercise_numbers = []
    
    # Extract regions between consecutive exercise markers
    for i in range(len(exercise_positions)):
        exercise_num, y_start = exercise_positions[i]
        
        # Determine end of current exercise
        if i < len(exercise_positions) - 1:
            y_end = exercise_positions[i+1][1]
        else:
            y_end = image.shape[0]  # End of image
        
        # Add some margin
        y_start = max(0, y_start - 10)
        y_end = min(image.shape[0], y_end + 10)
        
        # Extract the exercise image
        exercise_img = image[y_start:y_end, :]
        
        if exercise_img.size > 0:
            exercises.append(exercise_img)
            exercise_numbers.append(exercise_num)
    
    return exercises, exercise_numbers

def extract_text_from_image(image):
    """Extract text from an exercise image using improved OCR techniques"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing to improve OCR accuracy
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # OCR with multiple language support and page segmentation mode
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    return text

def is_duplicate(img1, img2, threshold=8):
    """Check if two images are duplicates based on perceptual hash"""
    try:
        hash1 = phash(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
        hash2 = phash(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)))
        return hash1 - hash2 < threshold
    except Exception as e:
        print(f"Error comparing images: {e}")
        return False

def deduplicate_exercises(exercises, numbers):
    """Remove duplicate exercises"""
    unique_exercises = []
    unique_numbers = []
    
    for i, exercise in enumerate(exercises):
        if not any(is_duplicate(exercise, unique) for unique in unique_exercises):
            unique_exercises.append(exercise)
            unique_numbers.append(numbers[i])
    
    return unique_exercises, unique_numbers

def get_ai_solution(exercise_text, client, language='Spanish'):
    """Get solution from AI for a given exercise text"""
    if language == 'Spanish':
        prompt = f"""
        Por favor, resuelve el siguiente ejercicio. Proporciona una solución detallada
        paso a paso y asegúrate de incluir la respuesta final.
        
        Ejercicio:
        {exercise_text}
        
        Solución:
        """
    if language == 'English':
        prompt = f"""
        Please solve the following exercise. Provide a detailed step-by-step solution
        and make sure to include the final answer.
        
        Exercise:
        {exercise_text}
        
        Solution:
        """

    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error obteniendo solución: {str(e)}"

def process_exams(folder_path, output_folder="exercises_results", language='Spanish'):
    """Process all exam images in a folder"""
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize AI client
    client = Together(api_key=TOGETHER_API_KEY)
    
    all_exercises = []
    all_numbers = []
    
    # First, extract all exercises from all exam images
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            
            exercises, numbers = extract_exercises(image_path)
            all_exercises.extend(exercises)
            all_numbers.extend(numbers)
            
            print(f"  Extraídos {len(exercises)} ejercicios")
    
    # Deduplicate exercises
    unique_exercises, unique_numbers = deduplicate_exercises(all_exercises, all_numbers)
    print(f"\nTotal de ejercicios únicos encontrados: {len(unique_exercises)}")
    
    # Create PDF report
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from io import BytesIO
    
    pdf_path = os.path.join(output_folder, "informe_ejercicios.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Create individual files and get AI solutions for each exercise
    for i, (exercise, number) in enumerate(zip(unique_exercises, unique_numbers)):
        print(f"\nProcesando Ejercicio {number} ({i+1}/{len(unique_exercises)})")
        
        # Save individual exercise image
        exercise_path = os.path.join(output_folder, f"ejercicio_{number}.jpg")
        cv2.imwrite(exercise_path, exercise)
        print(f"  Imagen guardada en {exercise_path}")
        
        # Extract text
        exercise_text = extract_text_from_image(exercise)
        text_path = os.path.join(output_folder, f"ejercicio_{number}_texto.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(exercise_text)
        print(f"  Texto extraído y guardado")
        
        # Get AI solution
        print(f"  Obteniendo solución con IA...")
        solution = get_ai_solution(exercise_text, client, language)
        solution_path = os.path.join(output_folder, f"ejercicio_{number}_solucion.txt")
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(solution)
        print(f"  Solución guardada en {solution_path}")
        
        # Add to PDF report
        if i > 0 and i % 2 == 0:  # New page every 2 exercises
            c.showPage()
        
        # Calculate positioning
        y_pos = height - 100 if i % 2 == 0 else height - 450
        
        # Add exercise number and image
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos + 50, f"Ejercicio {number}")
        
        # Convert OpenCV image to format usable by ReportLab
        img_bytes = BytesIO()
        pil_img = Image.fromarray(cv2.cvtColor(exercise, cv2.COLOR_BGR2RGB))
        
        # Resize if too large
        if pil_img.width > width - 100:
            ratio = (width - 100) / pil_img.width
            new_width = int(pil_img.width * ratio)
            new_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((new_width, new_height))
        
        pil_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Draw image
        img = ImageReader(img_bytes)
        img_width, img_height = pil_img.size
        c.drawImage(img, 50, y_pos - img_height, width=img_width, height=img_height)
        
        # Add solution (truncated if needed)
        c.setFont("Helvetica", 10)
        solution_lines = solution.split('\n')
        line_height = 12
        max_lines = 8
        
        c.drawString(50, y_pos - img_height - 20, "Solución:")
        for j, line in enumerate(solution_lines[:max_lines]):
            if len(line) > 80:
                line = line[:77] + "..."
            c.drawString(50, y_pos - img_height - 20 - (j+1)*line_height, line)
        
        if len(solution_lines) > max_lines:
            c.drawString(50, y_pos - img_height - 20 - (max_lines+1)*line_height,solution)
    
    c.save()
    print(f"\nInforme PDF guardado en {pdf_path}")
    print(f"Proceso completado. Todos los ejercicios y soluciones guardados en '{output_folder}'")

if __name__ == "__main__":
    print("Iniciando extracción y análisis de ejercicios de exámenes...")
    process_exams("models/", output_folder="exercises_results", language='Spanish')
    print("Proceso finalizado.")