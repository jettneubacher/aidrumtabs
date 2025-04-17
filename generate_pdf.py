from fpdf import FPDF
import os

# Desired vertical tab order
DRUM_LABELS = ["Cymbals", "Hi-Hat", "Snare", "Toms", "Kick"]
DRUM_SYMBOLS = ["X", "X", "O", "O", "O"]
# Mapping from display line to index in the hit_vector
HIT_VECTOR_INDEXES = [4, 3, 1, 2, 0]  # cymbal, hihat, snare, tom, kick

def generate_pdf(predictions, output_path, hits_per_row=32):
    """
    Generates a vertically accurate drum tab PDF in portrait format.

    Args:
        predictions (list of [float, list[int]]): List of [timestamp, [0/1 drum hits]].
        output_path (str): Full output PDF file path.
        hits_per_row (int): Number of timeframes (columns) per row in the PDF.

    Returns:
        str: Full path to the generated PDF.
    """
    pdf = FPDF(orientation='P', unit='mm', format='A4')  # Portrait layout
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=12)
    bold_font = ("Courier", "B", 12)
    normal_font = ("Courier", "", 12)

    '''
    # Smart estimate for hits per row
    page_width = 210  # A4 in mm (portrait)
    left_margin = pdf.l_margin
    right_margin = pdf.r_margin
    available_width = page_width - left_margin - right_margin

    label_width = pdf.get_string_width("Cymbals |")
    hit_width = pdf.get_string_width("X ")  # One character and a space
    hits_per_row = int((available_width - label_width) / hit_width)

    hits_per_row = max(1, hits_per_row - 4)
    '''
    hits_per_row = 24

    current_row = 0

    total_timeframes = len(predictions)

    for start in range(0, total_timeframes, hits_per_row):
        end = min(start + hits_per_row, total_timeframes)
        segment = predictions[start:end]

        if current_row == 4:
            pdf.add_page()
            current_row = 0

        for display_index in range(len(DRUM_LABELS)):
            drum_index = HIT_VECTOR_INDEXES[display_index]
            symbol = DRUM_SYMBOLS[display_index]
            line = ""

            for i, (_, hit_vector) in enumerate(segment):
                if hit_vector[drum_index]:
                    pdf.set_text_color(200, 0, 0)  # Red hit
                    line += symbol
                else:
                    pdf.set_text_color(150, 150, 150)  # Gray dash
                    line += "-"
                pdf.set_text_color(0, 0, 0)
                if (i + 1) % 4 == 0:
                    line += " | "
                else:
                    line += " "

            pdf.set_font(*bold_font)
            label = f"{DRUM_LABELS[display_index]:<8}| "
            pdf.set_font(*normal_font)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 10, txt=label + line.strip(), ln=1)

        # Divider after each full row of tabs
        pdf.cell(0, 5, txt="-" * 90, ln=1)
        pdf.ln(2)

        current_row += 1 #track rows printed on page so far

    pdf.output(output_path)
    return output_path
