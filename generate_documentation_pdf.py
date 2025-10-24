#!/usr/bin/env python3
"""
PDF Documentation Generator for Sehen Lernen Platform
Generates comprehensive feature documentation with identical formatting
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, blue, green
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
import os

def create_sehen_lernen_documentation():
    """Generate comprehensive Sehen Lernen documentation PDF"""
    
    # Setup PDF document
    filename = f"Sehen_Lernen_Comprehensive_Guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles matching the original formatting
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=HexColor('#2c3e50'),
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=20,
        textColor=HexColor('#2c3e50'),
        leftIndent=0
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=15,
        textColor=HexColor('#3498db'),
        leftIndent=0
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        textColor=HexColor('#2c3e50'),
        leftIndent=0
    )
    
    h4_style = ParagraphStyle(
        'CustomH4',
        parent=styles['Heading4'],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=10,
        textColor=HexColor('#34495e'),
        leftIndent=0
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        textColor=black,
        alignment=TA_JUSTIFY,
        leftIndent=0
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=4,
        textColor=black,
        leftIndent=20,
        bulletIndent=10
    )
    
    definition_style = ParagraphStyle(
        'Definition',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=12,
        textColor=HexColor('#27ae60'),
        leftIndent=20,
        rightIndent=20,
        borderColor=HexColor('#27ae60'),
        borderWidth=1,
        borderPadding=8,
        backColor=HexColor('#f8fff8')
    )
    
    # Story content
    story = []
    
    # Title page
    story.append(Paragraph("📋 Comprehensive Feature Guide", title_style))
    story.append(Paragraph("Sehen Lernen Platform", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<i>Complete with Technical Term Definitions</i>", styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Project info
    project_info = f"""
    <b>Project:</b> Sehen Lernen: Human and AI Comparison<br/>
    <b>Funded by:</b> Innovation in Teaching Foundation (Freedom 2023 program)<br/>
    <b>Contact:</b> Prof. Dr. Martin Langner<br/>
    <b>Team:</b> Luana Moraes Costa, Firmin Forster, M. Kipke, Alexander Eric Wilhelm, Alexander Zeckey<br/>
    <b>Institution:</b> Institut für Digital Humanities, Georg-August-Universität Göttingen<br/>
    <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}
    """
    story.append(Paragraph(project_info, styles['Normal']))
    story.append(PageBreak())
    
    # Home Page & Navigation
    story.append(Paragraph("🏠 Home Page & Navigation", h1_style))
    home_items = [
        "🚀 <b>Start Learning Button:</b> Entry point to begin image analysis journey",
        "🔍 <b>Visual Analysis Icon:</b> Overview of visual processing capabilities",  
        "🧠 <b>AI Comparison Icon:</b> Human vs AI comparison features",
        "📊 <b>Interactive Learning Icon:</b> Educational tools and insights"
    ]
    for item in home_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    
    story.append(Spacer(1, 20))
    
    # Data Input & Upload
    story.append(Paragraph("📂 Data Input & Upload (Sidebar)", h1_style))
    
    story.append(Paragraph("Upload Methods", h2_style))
    upload_items = [
        "📁 <b>Upload Images:</b> Direct file upload for PNG, JPG, JPEG files",
        "📦 <b>Upload ZIP:</b> Bulk upload via compressed archive containing multiple images",
        "📄 <b>Extract from CSV:</b> Automatically download images from a list of URLs stored in a CSV file"
    ]
    for item in upload_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    
    story.append(Paragraph("Upload Controls", h2_style))
    control_items = [
        "<b>Upload Images Button:</b> Process selected files and send them to the backend server for analysis",
        "<b>Extract Images from CSV Button:</b> Download images from URL list and add to dataset",
        "<b>Remove All Images:</b> Clear entire image dataset from memory and storage",
        "<b>Remove Individual Images:</b> Delete specific images from the current collection"
    ]
    for item in control_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    
    story.append(PageBreak())
    
    # Feature Selection Hub
    story.append(Paragraph("🔧 Feature Selection (Main Analysis Hub)", h1_style))
    
    # Tab 1: Histogram Analysis
    story.append(Paragraph("Tab 1: Histogram Analysis 📊", h2_style))
    story.append(Paragraph("<b>Purpose:</b> Analyze color and brightness distribution patterns in images", body_style))
    
    histogram_def = """
    <b>📖 What is a Histogram?</b><br/>
    A histogram is a graph showing how frequently different brightness values (or colors) appear in an image. 
    Think of it as a "color census" - it counts how many pixels have each brightness level from black (0) to white (255).
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(histogram_def, definition_style))
    
    story.append(Paragraph("Controls:", h3_style))
    hist_controls = [
        "<b>Histogram Type Radio:</b><br/>• <i>Black and White:</i> Analyzes brightness levels in grayscale images<br/>• <i>Colored:</i> Analyzes color channels (Red, Green, Blue) separately",
        "<b>Generate histogram for all images:</b> Process entire dataset at once for comparison",
        "<b>Select image:</b> Choose single image for detailed analysis",
        "<b>Compute Histogram:</b> Generate color/brightness distribution charts",
        "<b>Download All Histograms:</b> Export numerical histogram data as CSV for further analysis"
    ]
    for control in hist_controls:
        story.append(Paragraph(f"• {control}", bullet_style))
    
    story.append(Spacer(1, 15))
    
    # Tab 2: K-means Clustering
    story.append(Paragraph("Tab 2: K-means Clustering 🎯", h2_style))
    story.append(Paragraph("<b>Purpose:</b> Automatically group similar pixels or images together", body_style))
    
    kmeans_def = """
    <b>📖 What is K-means Clustering?</b><br/>
    K-means is an algorithm that divides data into K groups (clusters) by similarity. In images, it can:<br/>
    • <b>Pixel Clustering:</b> Group pixels with similar colors (useful for color quantization/simplification)<br/>
    • <b>Image Clustering:</b> Group entire images with similar visual characteristics<br/>
    • <b>How it works:</b> The algorithm finds K "center points" and assigns each data point to the closest center, iteratively improving the grouping
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(kmeans_def, definition_style))
    
    story.append(Paragraph("Controls:", h3_style))
    kmeans_controls = [
        "<b>Clustering Mode:</b><br/>• <i>Single Image:</i> Cluster pixels within one image (color segmentation)<br/>• <i>Multi-Image Clustering:</i> Group multiple images by visual similarity",
        "<b>Number of Clusters (K):</b> Slider (2-20) - How many groups to create",
        "<b>Random Seed:</b> Number ensuring reproducible results (same seed = same clusters every time)",
        "<b>Image Selection:</b> Choose which images to include in clustering analysis",
        "<b>Select all images:</b> Process entire dataset",
        "<b>Perform K-means:</b> Execute clustering algorithm",
        "<b>Download Cluster Data:</b> Export clustering results and group assignments"
    ]
    for control in kmeans_controls:
        story.append(Paragraph(f"• {control}", bullet_style))
    
    story.append(PageBreak())
    
    # Tab 3: Advanced Feature Extraction
    story.append(Paragraph("Tab 3: Advanced Feature Extraction 🔬", h2_style))
    story.append(Paragraph("<b>Purpose:</b> Extract sophisticated mathematical descriptions of image content", body_style))
    
    # CNN Embeddings
    story.append(Paragraph("CNN Embedding Extraction", h3_style))
    cnn_def = """
    <b>📖 What are CNN Embeddings?</b><br/>
    CNN (Convolutional Neural Network) embeddings are mathematical representations of images created by deep learning models. 
    Think of them as "digital DNA" - a unique numerical fingerprint that captures what the AI "sees" in the image (objects, textures, patterns).
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(cnn_def, definition_style))
    
    cnn_controls = [
        "<b>Model Selection:</b> Choose CNN architecture:<br/>• <i>ResNet:</i> Residual Network - excellent for object recognition<br/>• <i>VGG:</i> Visual Geometry Group model - good for detailed features<br/>• <i>AlexNet:</i> Classic CNN - fast but less detailed",
        "<b>Extract Embeddings:</b> Generate deep learning feature vectors (typically 512-2048 numbers per image)",
        "<b>Download All Embeddings:</b> Export feature vectors as CSV for machine learning"
    ]
    for control in cnn_controls:
        story.append(Paragraph(f"• {control}", bullet_style))
    
    # LBP
    story.append(Paragraph("LBP (Local Binary Pattern)", h3_style))
    lbp_def = """
    <b>📖 What is LBP?</b><br/>
    LBP analyzes local texture patterns by comparing each pixel with its neighbors. It creates a "texture code" by checking 
    if neighboring pixels are brighter or darker, forming binary patterns that describe surface textures like fabric weaves, bark patterns, or skin textures.
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(lbp_def, definition_style))
    
    lbp_controls = [
        "<b>Radius:</b> How far to look for neighboring pixels (1-16 pixels away)",
        "<b>Number of Neighbors:</b> How many surrounding pixels to compare (8, 16, or 24 points)",
        "<b>Method:</b> Algorithm variant:<br/>• <i>default:</i> Basic LBP calculation<br/>• <i>ror:</i> Rotation-invariant (same result regardless of image rotation)<br/>• <i>uniform:</i> Only considers uniform patterns (most common textures)<br/>• <i>var:</i> Variance-based (captures texture contrast)",
        "<b>Normalize histogram:</b> Scale results to 0-1 range for comparison",
        "<b>Compute LBP:</b> Generate texture pattern analysis",
        "<b>Download LBP Data:</b> Export texture feature measurements"
    ]
    for control in lbp_controls:
        story.append(Paragraph(f"• {control}", bullet_style))
    
    story.append(PageBreak())
    
    # Haralick Features
    story.append(Paragraph("Haralick Texture Features (GLCM)", h3_style))
    haralick_def = """
    <b>📖 What are Haralick Features?</b><br/>
    Haralick features describe image texture using GLCM (Gray-Level Co-occurrence Matrix). GLCM counts how often pixel pairs 
    with specific brightness values occur at certain distances and angles. It measures texture properties like:<br/>
    • <i>Contrast:</i> How much brightness varies<br/>
    • <i>Homogeneity:</i> How uniform the texture is<br/>
    • <i>Energy:</i> How orderly the texture appears<br/>
    • <i>Correlation:</i> How predictable pixel relationships are
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(haralick_def, definition_style))
    
    haralick_controls = [
        "<b>Quantization Levels:</b> How many gray levels to consider (8-256) - fewer levels = faster computation",
        "<b>Distances:</b> Pixel separation distances to analyze (e.g., 1, 2, 3 pixels apart)",
        "<b>Angles:</b> Directional analysis:<br/>• <i>0°:</i> Horizontal patterns<br/>• <i>45°:</i> Diagonal patterns (top-right to bottom-left)<br/>• <i>90°:</i> Vertical patterns<br/>• <i>135°:</i> Diagonal patterns (top-left to bottom-right)",
        "<b>Properties:</b> Texture measurements to calculate (contrast, homogeneity, energy, correlation, ASM, dissimilarity)",
        "<b>Resize Options:</b> Preprocessing to standardize image sizes",
        "<b>Compute Haralick:</b> Generate texture statistics",
        "<b>Download Texture Data:</b> Export GLCM-based measurements"
    ]
    for control in haralick_controls:
        story.append(Paragraph(f"• {control}", bullet_style))
    
    # Contour Extraction
    story.append(Paragraph("Contour Extraction", h3_style))
    contour_def = """
    <b>📖 What are Contours?</b><br/>
    Contours are curves that outline object boundaries in images. Think of them as "digital tracing" - they follow the edges 
    where objects meet backgrounds or where different objects touch each other.
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(contour_def, definition_style))
    
    contour_controls = [
        "<b>Mode:</b> Contour detection method:<br/>• <i>RETR_EXTERNAL:</i> Only outermost contours<br/>• <i>RETR_LIST:</i> All contours without hierarchy<br/>• <i>RETR_TREE:</i> All contours with parent-child relationships",
        "<b>Approximation:</b> Shape simplification method:<br/>• <i>CHAIN_APPROX_SIMPLE:</i> Compress contours (remove redundant points)<br/>• <i>CHAIN_APPROX_NONE:</i> Keep all contour points",
        "<b>Min Area:</b> Filter out small contours (noise reduction)",
        "<b>Return Bounding Boxes:</b> Include rectangular boxes around each contour",
        "<b>Extract Contours:</b> Find and trace object boundaries"
    ]
    for control in contour_controls:
        story.append(Paragraph(f"• {control}", bullet_style))
    
    story.append(PageBreak())
    
    # Tab 4: Similarity Search
    story.append(Paragraph("Tab 4: Similarity Search 🔍", h2_style))
    story.append(Paragraph("<b>Purpose:</b> Find visually similar images using advanced computer vision algorithms", body_style))
    
    story.append(Paragraph("Search Modes:", h3_style))
    search_modes = [
        "<b>Single vs All:</b> Select one query image and find all similar images in dataset",
        "<b>Pairwise:</b> Compare exactly two specific images for similarity score"
    ]
    for mode in search_modes:
        story.append(Paragraph(f"• {mode}", bullet_style))
    
    story.append(Paragraph("📖 Feature Methods Explained:", h3_style))
    
    story.append(Paragraph("CNN (Convolutional Neural Network)", h4_style))
    story.append(Paragraph("Deep learning approach that mimics human visual processing. Analyzes images in layers, detecting edges, then shapes, then complex objects. Best general-purpose method for most image types.", body_style))
    
    story.append(Paragraph("HOG (Histogram of Oriented Gradients)", h4_style))
    story.append(Paragraph("Analyzes the direction and strength of edges in image regions. Excellent for detecting shapes and objects because it captures how brightness changes across the image. Commonly used in pedestrian detection and object recognition.", body_style))
    
    story.append(Paragraph("SIFT (Scale-Invariant Feature Transform)", h4_style))
    story.append(Paragraph("Finds distinctive \"keypoints\" in images - unique spots that remain recognizable even if the image is rotated, scaled, or viewed from different angles. Like finding landmarks that are easy to spot from any direction.", body_style))
    
    story.append(Paragraph("Histogram Similarity", h4_style))
    story.append(Paragraph("Compares color distribution patterns. Images with similar color palettes will be considered similar, regardless of spatial arrangement. Good for finding images with matching color schemes.", body_style))
    
    story.append(Paragraph("Manuscript (Specialized)", h4_style))
    manuscript_desc = """
    Custom algorithm designed for historical documents, ancient texts, and handwritten materials. Combines multiple techniques optimized for document analysis:
    • <b>CLAHE:</b> Contrast enhancement for faded texts
    • <b>Multi-scale HOG:</b> Analyzes both fine details (letters) and coarse patterns (layout)
    • <b>Layout analysis:</b> Understands text line and word structure
    """
    story.append(Paragraph(manuscript_desc, body_style))
    
    story.append(Paragraph("Distance Metrics:", h3_style))
    metrics = [
        "<b>Cosine:</b> Measures angle between feature vectors (0 = identical, 1 = completely different)",
        "<b>Euclidean:</b> Straight-line distance in feature space (like measuring with a ruler)",
        "<b>Manhattan:</b> Grid-based distance (like city blocks - only horizontal/vertical moves)"
    ]
    for metric in metrics:
        story.append(Paragraph(f"• {metric}", bullet_style))
    
    story.append(PageBreak())
    
    # Tab 5: SIFT & Edge Detection
    story.append(Paragraph("Tab 5: SIFT & Edge Detection 🎯", h2_style))
    story.append(Paragraph("<b>Purpose:</b> Detect distinctive points and boundaries in images", body_style))
    
    sift_def = """
    <b>📖 SIFT Explained (Scale-Invariant Feature Transform):</b><br/>
    SIFT finds "interest points" - distinctive spots in images that can be reliably found again even if the image is rotated, 
    resized, or viewed from different angles. Each keypoint gets a 128-number "descriptor" that uniquely identifies its local appearance.
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(sift_def, definition_style))
    
    edge_def = """
    <b>📖 Edge Detection:</b><br/>
    Edge detection finds boundaries between different regions in images - places where brightness changes rapidly. 
    Edges reveal object shapes, boundaries, and structural elements.
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(edge_def, definition_style))
    
    # Tab 6: Classifier Training
    story.append(Paragraph("Tab 6: Classifier Training 🤖", h2_style))
    story.append(Paragraph("<b>Purpose:</b> Train machine learning models to automatically categorize images", body_style))
    
    classifier_def = """
    <b>📖 What is Image Classification?</b><br/>
    Machine learning technique that teaches computers to recognize and categorize images. You provide labeled examples (training data), 
    and the algorithm learns patterns to classify new, unseen images.
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(classifier_def, definition_style))
    
    story.append(Paragraph("Workflow:", h3_style))
    workflow_steps = [
        "<b>Label Assignment:</b> Manually assign category names to training images (e.g., \"cat\", \"dog\", \"car\")",
        "<b>Train/Test Split:</b> Divide data into training set (to learn from) and test set (to evaluate accuracy)",
        "<b>Feature Selection:</b> Choose how to represent images mathematically:<br/>• <i>HOG:</i> Shape and edge features<br/>• <i>LBP:</i> Texture patterns<br/>• <i>Embeddings:</i> Deep learning features",
        "<b>Model Training:</b> Algorithm learns from labeled examples",
        "<b>Prediction:</b> Classify new images based on learned patterns"
    ]
    for step in workflow_steps:
        story.append(Paragraph(f"{len([s for s in workflow_steps if workflow_steps.index(s) <= workflow_steps.index(step)])}. {step}", bullet_style))
    
    story.append(PageBreak())
    
    # Statistics and Visualization
    story.append(Paragraph("📊 Statistics Analysis", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Perform statistical analysis on extracted features and results", body_style))
    
    stats_features = [
        "<b>Download Result:</b> Export statistical analysis results",
        "<b>Download Settings:</b> Save analysis configuration for reproducibility",
        "<b>Next: Visualization:</b> Navigate to data visualization tools"
    ]
    for feature in stats_features:
        story.append(Paragraph(f"• {feature}", bullet_style))
    
    story.append(Paragraph("📈 Visualization", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Create visual representations of analysis results", body_style))
    
    viz_features = [
        "<b>Download Result:</b> Export visualization data and charts",
        "<b>Download Settings:</b> Save visualization parameters",
        "<b>Previous: Statistical Analysis:</b> Navigate back to statistics"
    ]
    for feature in viz_features:
        story.append(Paragraph(f"• {feature}", bullet_style))
    
    # Seher AI Assistant
    story.append(Paragraph("💬 Seher AI Assistant", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Intelligent help system with natural language understanding", body_style))
    
    seher_def = """
    <b>📖 What is Seher?</b><br/>
    Seher is an AI chatbot that understands natural language questions about computer vision and provides contextual, 
    educational responses. It can explain complex algorithms in simple terms and guide users through the platform.
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(seher_def, definition_style))
    
    seher_features = [
        "<b>💬 Toggle Button:</b> Open/close chat interface",
        "<b>Natural Language Queries:</b> Ask questions in plain English",
        "<b>Contextual Responses:</b> Answers adapt based on current page/feature",
        "<b>Technical Explanations:</b> Detailed algorithm descriptions with analogies",
        "<b>Usage Guidance:</b> Step-by-step instructions for complex tasks",
        "<b>Troubleshooting:</b> Error resolution and debugging assistance"
    ]
    for feature in seher_features:
        story.append(Paragraph(f"• {feature}", bullet_style))
    
    story.append(Paragraph("Example Queries:", h3_style))
    example_queries = [
        "\"What's the difference between CNN and SIFT?\"",
        "\"When should I use K-means clustering?\"",
        "\"How do I interpret similarity scores?\"",
        "\"What parameters work best for texture analysis?\"",
        "\"Help me choose the right feature extraction method\""
    ]
    for query in example_queries:
        story.append(Paragraph(f"• {query}", bullet_style))
    
    story.append(PageBreak())
    
    # Technical Terms Summary
    story.append(Paragraph("🎯 Key Technical Terms Summary", h1_style))
    
    story.append(Paragraph("Core Concepts", h2_style))
    core_concepts = [
        "<b>Feature Extraction:</b> Converting images into numerical data for analysis",
        "<b>Machine Learning:</b> Algorithms that learn patterns from data",
        "<b>Computer Vision:</b> Teaching computers to understand and interpret images",
        "<b>Neural Network:</b> AI system inspired by how brain neurons work"
    ]
    for concept in core_concepts:
        story.append(Paragraph(f"• {concept}", bullet_style))
    
    story.append(Paragraph("Algorithms", h2_style))
    algorithms = [
        "<b>K-means:</b> Grouping algorithm that finds clusters in data",
        "<b>CNN:</b> Deep learning network that processes images in layers",
        "<b>SIFT:</b> Keypoint detector that finds distinctive image features",
        "<b>HOG:</b> Edge and shape detector using gradient analysis",
        "<b>LBP:</b> Texture analyzer using local pixel patterns",
        "<b>Haralick:</b> Texture measurement using spatial relationships (GLCM)"
    ]
    for algorithm in algorithms:
        story.append(Paragraph(f"• {algorithm}", bullet_style))
    
    story.append(Paragraph("Measurements", h2_style))
    measurements = [
        "<b>Similarity Score:</b> Number indicating how alike two images are (0-1 scale)",
        "<b>Distance Metric:</b> Mathematical method for measuring differences",
        "<b>Feature Vector:</b> List of numbers representing image characteristics",
        "<b>Clustering:</b> Grouping similar items together automatically"
    ]
    for measurement in measurements:
        story.append(Paragraph(f"• {measurement}", bullet_style))
    
    # Conclusion
    story.append(Spacer(1, 30))
    conclusion = """
    This comprehensive platform combines <b>computer vision</b>, <b>machine learning</b>, <b>pattern recognition</b>, and 
    <b>statistical analysis</b> into an educational tool that makes advanced image analysis accessible to users at all technical levels! 🚀
    """
    story.append(Paragraph(conclusion, body_style))
    
    # Build PDF
    doc.build(story)
    
    return filename

if __name__ == "__main__":
    try:
        filename = create_sehen_lernen_documentation()
        print(f"✅ PDF documentation generated successfully!")
        print(f"📄 Filename: {filename}")
        print(f"📁 Location: {os.path.abspath(filename)}")
        print(f"🎯 The PDF maintains identical formatting with:")
        print("   • Emojis and icons")
        print("   • Color-coded headers")
        print("   • Definition boxes")
        print("   • Bullet points and structure")
        print("   • Technical term explanations")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("💡 Make sure you have reportlab installed: pip install reportlab")