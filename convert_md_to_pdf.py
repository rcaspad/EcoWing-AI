#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import re
from datetime import datetime

# Configuración
pdf_file = "website/assets/Whitepaper_EcoWing_AI.pdf"
md_file = "website/assets/Whitepaper_EcoWing.md"

# Crear documento PDF
doc = SimpleDocTemplate(pdf_file, pagesize=A4,
                       rightMargin=1.5*cm, leftMargin=1.5*cm,
                       topMargin=1.5*cm, bottomMargin=1.5*cm)

# Estilos personalizados
styles = getSampleStyleSheet()
story = []

# Colores personalizados
NEON_GREEN = colors.HexColor("#39ff14")
NEON_ORANGE = colors.HexColor("#ff6b35")
DARK_BG = colors.HexColor("#0a0a0a")

# Estilos personalizados
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=28,
    textColor=NEON_GREEN,
    spaceAfter=30,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading1_style = ParagraphStyle(
    'CustomHeading1',
    parent=styles['Heading1'],
    fontSize=18,
    textColor=NEON_ORANGE,
    spaceAfter=12,
    spaceBefore=12,
    fontName='Helvetica-Bold'
)

heading2_style = ParagraphStyle(
    'CustomHeading2',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=NEON_GREEN,
    spaceAfter=8,
    spaceBefore=8,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['BodyText'],
    fontSize=11,
    textColor=colors.HexColor("#ffffff"),
    spaceAfter=10,
    alignment=TA_JUSTIFY,
    fontName='Helvetica'
)

# Leer archivo Markdown
try:
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: No se encontró {md_file}")
    exit(1)

# Procesar contenido Markdown simple
lines = content.split('\n')
in_code_block = False
table_data = []
current_text = ""

for line in lines:
    line = line.strip()
    
    # Saltar líneas vacías entre párrafos
    if not line:
        if current_text:
            story.append(Paragraph(current_text, body_style))
            story.append(Spacer(1, 0.2*cm))
            current_text = ""
        continue
    
    # Títulos
    if line.startswith('# '):
        if current_text:
            story.append(Paragraph(current_text, body_style))
            current_text = ""
        story.append(Paragraph(line.replace('# ', ''), title_style))
        story.append(Spacer(1, 0.3*cm))
    
    # Subtítulos
    elif line.startswith('## '):
        if current_text:
            story.append(Paragraph(current_text, body_style))
            current_text = ""
        story.append(Paragraph(line.replace('## ', ''), heading1_style))
        story.append(Spacer(1, 0.2*cm))
    
    # Sub-subtítulos
    elif line.startswith('### '):
        if current_text:
            story.append(Paragraph(current_text, body_style))
            current_text = ""
        story.append(Paragraph(line.replace('### ', ''), heading2_style))
        story.append(Spacer(1, 0.1*cm))
    
    # Listas
    elif line.startswith('- '):
        if current_text:
            story.append(Paragraph(current_text, body_style))
            current_text = ""
        bullet_text = "• " + line.replace('- ', '')
        story.append(Paragraph(bullet_text, body_style))
        story.append(Spacer(1, 0.1*cm))
    
    # Código
    elif line.startswith('```'):
        in_code_block = not in_code_block
    
    # Contenido normal
    else:
        if not in_code_block:
            # Reemplazar formatos Markdown simples
            line = line.replace('**', '')
            line = line.replace('*', '')
            line = line.replace('`', '')
            
            if current_text:
                current_text += " " + line
            else:
                current_text = line

# Agregar último párrafo si existe
if current_text:
    story.append(Paragraph(current_text, body_style))

# Pie de página con metadata
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("—" * 80, body_style))
story.append(Spacer(1, 0.2*cm))

footer_text = f"<b>Documento generado el {datetime.now().strftime('%d de %B de %Y')}</b><br/>Versión 1.0 - EcoWing AI MVP<br/>www.ecowing.ai"
footer_style = ParagraphStyle(
    'Footer',
    parent=styles['Normal'],
    fontSize=9,
    textColor=colors.HexColor("#999999"),
    alignment=TA_CENTER
)
story.append(Paragraph(footer_text, footer_style))

# Construir PDF
try:
    doc.build(story)
    print(f"✅ PDF creado exitosamente: {pdf_file}")
except Exception as e:
    print(f"❌ Error al crear PDF: {e}")
    exit(1)
