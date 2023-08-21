from pathlib import Path


def generate_html_report(face_image_paths: list, output_html: Path) -> None:
    """
    Generate an HTML report showcasing the detected faces.

    Args:
        face_image_paths (list): List of paths to the detected face images.
        output_html (str): Path to save the HTML report.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detected Faces Report</title>
        <style>
            img {
                margin: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                width: 150px;
            }
        </style>
    </head>
    <body>
        <h2>Detected Faces</h2>
        {face_images}
    </body>
    </html>
    """

    face_image_tags = "".join(
        [f'<img src="{path}" alt="Detected Face {idx}" />' for idx, path in enumerate(face_image_paths)])
    with open(output_html.as_posix(), 'w') as file:
        file.write(html_template.replace("{face_images}", face_image_tags))
