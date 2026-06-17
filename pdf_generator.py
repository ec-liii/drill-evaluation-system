import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.dimgray)
        self.setLineWidth(0.5)
        self.setStrokeColor(colors.lightgrey)
        
        # Safely capture dynamic letter coordinates
        width, height = self._pagesize
        self.line(36, 45, width - 36, 45)
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(width - 36, 30, page_text)
        self.drawString(36, 30, "CONFIDENTIAL - MILITARY DRILL EVALUATION SYSTEM")
        self.restoreState()

def generate_drill_pdf(session_data, snapshot_img_path, output_pdf_path, soldier_name="Cadet Kumar", soldier_id="NCC/2026/8492"):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=40, bottomMargin=60)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#8B0000'), alignment=1, spaceAfter=15)
    section_style = ParagraphStyle('SecTitle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#1A365D'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#2D3748'))
    bold_body = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')
    error_style = ParagraphStyle('ErrorText', parent=body_style, textColor=colors.HexColor('#C53030'), fontName='Helvetica-Bold')

    drill_name = session_data[0]['drill_type'].replace('_', ' ').title() if session_data else "Unknown Drill"
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    errors_encountered = list(set([d['metrics']['error_flag'] for d in session_data if d['metrics'] and d['metrics'].get('error_flag')]))
    final_score = max(100 - (len(errors_encountered) * 10), 0)
    
    story.append(Paragraph("ARTIFICIAL INTELLIGENCE DRILL EVALUATION REPORT", title_style))
    story.append(Spacer(1, 10))
    
    metadata_data = [
        [Paragraph("<b>Soldier Name:</b>", body_style), Paragraph(soldier_name, body_style), Paragraph("<b>Evaluation Date:</b>", body_style), Paragraph(current_date, body_style)],
        [Paragraph("<b>Soldier ID:</b>", body_style), Paragraph(soldier_id, body_style), Paragraph("<b>Evaluation Time:</b>", body_style), Paragraph(current_time, body_style)],
        [Paragraph("<b>Drill Target:</b>", body_style), Paragraph(drill_name, body_style), Paragraph("<b>Final Performance Score:</b>", bold_body), Paragraph(f"<b>{final_score} / 100</b>", bold_body)]
    ]
    
    meta_table = Table(metadata_data, colWidths=[90, 180, 110, 160])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Logged Behavioral Deviations / Errors", section_style))
    if errors_encountered:
        error_table_content = [[Paragraph("<b>Detected Infraction Details</b>", bold_body)]]
        for err in errors_encountered:
            error_table_content.append([Paragraph(f"• {err}", error_style)])
            
        error_table = Table(error_table_content, colWidths=[540])
        error_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#FED7D7')),
            ('PADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#FEB2B2')),
        ]))
        story.append(error_table)
    else:
        story.append(Paragraph("<font color='#2F855A'><b>PASSED:</b> No drill deviations or posture errors detected.</font>", body_style))
    
    story.append(Spacer(1, 20))
    
    evidence_elements = []
    evidence_elements.append(Paragraph("Computer Vision Skeletal Evidence Logging", section_style))
    evidence_elements.append(Spacer(1, 5))
    
    if os.path.exists(snapshot_img_path):
        img = Image(snapshot_img_path, width=440, height=330)
        img.hAlign = 'CENTER'
        evidence_elements.append(img)
        evidence_elements.append(Spacer(1, 10))
        evidence_elements.append(Paragraph("<i>Figure 1.0: Frame displaying documented algorithmic coordinate deviation from target benchmarks.</i>", body_style))
        
    story.append(KeepTogether(evidence_elements))
    doc.build(story, canvasmaker=NumberedCanvas)