from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import os

def create_sample_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Invoice #12345", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("This is a test PDF containing both text and an image.", styles['Normal']))

    # assumes sample.jpg is in the same folder
    if os.path.exists("sample.jpg"):
        img = Image("sample.jpg", width=200, height=150)
        story.append(Spacer(1, 20))
        story.append(img)

    doc.build(story)
    print(f"✅ PDF created: {filename}")

if __name__ == "__main__":
    create_sample_pdf("storage/sample_files/test_invoice.pdf")
