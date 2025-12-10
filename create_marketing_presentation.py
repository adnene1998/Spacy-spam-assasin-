#!/usr/bin/env python3
"""
Script to create a marketing plan presentation for an event management platform.
Includes: SCP, 4Ps, Buyer Persona, and Ideal Buyer Journey.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Color constants for placeholder images
COLOR_PLATFORM = (52, 152, 219)  # Blue
COLOR_SEGMENTATION = (46, 204, 113)  # Green
COLOR_MARKETING = (155, 89, 182)  # Purple
COLOR_PERSONA_1 = (230, 126, 34)  # Orange
COLOR_PERSONA_2 = (231, 76, 60)  # Red
COLOR_JOURNEY = (52, 73, 94)  # Dark blue

# Image dimensions
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
PERSONA_IMAGE_SIZE = 350

def create_placeholder_image(text, filename, width=400, height=300, bg_color=(70, 130, 180), text_color=(255, 255, 255)):
    """Create a placeholder image with text
    
    Args:
        text: Text to display on the image
        filename: Filename to save the image
        width: Image width in pixels
        height: Image height in pixels
        bg_color: Background color RGB tuple
        text_color: Text color RGB tuple
    """
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a system font with cross-platform fallback
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 24)
                break
        except (OSError, IOError):
            continue
    
    # Use default font if no system font found
    if font is None:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    draw.text(position, text, fill=text_color, font=font)
    img.save(filename)
    return filename

def add_image_to_slide(slide, image_path, left, top, width, height):
    """Add an image to a slide
    
    Args:
        slide: PowerPoint slide object
        image_path: Path to the image file
        left: Left position in inches
        top: Top position in inches
        width: Image width in inches (None to maintain aspect ratio)
        height: Image height in inches (None to maintain aspect ratio)
    """
    if os.path.exists(image_path):
        if width and height:
            slide.shapes.add_picture(image_path, Inches(left), Inches(top), 
                                    width=Inches(width), height=Inches(height))
        else:
            slide.shapes.add_picture(image_path, Inches(left), Inches(top))

def create_title_slide(prs, title, subtitle):
    """Create a title slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    return slide

def create_content_slide(prs, title):
    """Create a content slide with title and content area"""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    return slide

def add_bullet_points(textbox, points, level=0):
    """Add bullet points to a text frame
    
    Args:
        textbox: PowerPoint textbox shape to add points to
        points: List of strings to add as bullet points
        level: Indentation level for bullets (default: 0)
    """
    text_frame = textbox.text_frame
    text_frame.clear()
    
    for i, point in enumerate(points):
        if i == 0:
            p = text_frame.paragraphs[0]
        else:
            p = text_frame.add_paragraph()
        
        p.text = point
        p.level = level
        p.font.size = Pt(18)

def create_marketing_presentation():
    """Create the complete marketing presentation
    
    Creates a comprehensive 15-slide PowerPoint presentation covering:
    - SCP (Segmentation, Targeting, Positioning)
    - The 4 Ps of Marketing (Product, Price, Place, Promotion)
    - Buyer Personas
    - Ideal Buyer Journey
    
    Returns:
        str: Filename of the created presentation
    """
    # Create placeholder images for the presentation
    images_dir = "presentation_images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Create images with different colors for different sections
    create_placeholder_image("Plateforme\nÉvénementiel", f"{images_dir}/platform.png", 
                           bg_color=COLOR_PLATFORM, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    create_placeholder_image("Segmentation\nMarché", f"{images_dir}/segmentation.png", 
                           bg_color=COLOR_SEGMENTATION, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    create_placeholder_image("Marketing Mix\n4P", f"{images_dir}/marketing_mix.png", 
                           bg_color=COLOR_MARKETING, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    create_placeholder_image("Thomas\n30 ans", f"{images_dir}/persona_thomas.png", 
                           bg_color=COLOR_PERSONA_1, width=PERSONA_IMAGE_SIZE, height=PERSONA_IMAGE_SIZE)
    create_placeholder_image("Marc\n38 ans", f"{images_dir}/persona_marc.png", 
                           bg_color=COLOR_PERSONA_2, width=PERSONA_IMAGE_SIZE, height=PERSONA_IMAGE_SIZE)
    create_placeholder_image("Parcours Client\nJourney", f"{images_dir}/customer_journey.png", 
                           bg_color=COLOR_JOURNEY, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Slide 1: Title
    slide = create_title_slide(
        prs,
        "Plan Marketing",
        "Plateforme de Gestion de Fêtes et Événements"
    )
    
    # Slide 2: Introduction - Vue d'ensemble
    slide = create_content_slide(prs, "Vue d'Ensemble du Projet")
    # Add image on the right side
    add_image_to_slide(slide, f"{images_dir}/platform.png", 6.5, 2, 3, 2.25)
    # Add text on the left side
    left = Inches(0.5)
    top = Inches(2)
    width = Inches(5.5)
    height = Inches(4)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Plateforme digitale de gestion d'événements",
        "Solution complète pour la planification et l'organisation de fêtes",
        "Public cible: Particuliers et professionnels de l'événementiel",
        "Objectif: Simplifier l'organisation d'événements mémorables"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 3: SCP - Segmentation
    slide = create_content_slide(prs, "SCP - Segmentation")
    # Add image on the right
    add_image_to_slide(slide, f"{images_dir}/segmentation.png", 6.5, 2.5, 3, 2.25)
    left = Inches(0.5)
    top = Inches(1.8)
    width = Inches(5.5)
    height = Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Processus de segmentation hiérarchique:",
        "",
        "1. Segmentation primaire par GENRE:",
        "  → Hommes | Femmes",
        "",
        "2. Segmentation HOMMES:",
        "  • Jeunes mariés (25-35 ans) - organisent leur mariage",
        "  • Professionnels événementiels (30-50 ans)",
        "  • Entrepreneurs/Entreprises - événements corporatifs",
        "",
        "3. Segmentation FEMMES:",
        "  • Jeunes mariées (25-35 ans) - organisent leur mariage",
        "  • Organisatrices familiales (30-50 ans) - fêtes familiales",
        "  • Professionnelles événementielles (25-55 ans)",
        "",
        "4. Segmentation secondaire (transversale):",
        "  • Par budget: Économique | Standard | Premium",
        "  • Par type d'événement: Personnel | Professionnel"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 4: SCP - Ciblage
    slide = create_content_slide(prs, "SCP - Ciblage")
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Cible Principale: Particuliers organisateurs d'événements",
        "  - 25-45 ans, urbains, connectés digitalement",
        "  - Recherchent des solutions clés en main",
        "  - Valorisent le gain de temps et la qualité",
        "Cible Secondaire: Organisateurs professionnels",
        "  - Agences événementielles de petite à moyenne taille",
        "  - Besoin d'optimiser leur workflow",
        "Critères de sélection:",
        "  - Potentiel de croissance élevé",
        "  - Adoption technologique forte",
        "  - Capacité de dépense confirmée"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 5: SCP - Positionnement
    slide = create_content_slide(prs, "SCP - Positionnement")
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Position: 'La plateforme tout-en-un pour des événements réussis'",
        "Proposition de valeur unique:",
        "  - Interface intuitive et conviviale",
        "  - Gestion complète de A à Z",
        "  - Réseau de prestataires vérifiés",
        "  - Outils de budgétisation et de suivi",
        "Avantages concurrentiels:",
        "  - Technologie innovante et moderne",
        "  - Service client réactif et personnalisé",
        "  - Tarification transparente et flexible",
        "Différenciation: Combinaison unique de simplicité d'utilisation et de fonctionnalités professionnelles"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 6: Les 4P - Produit
    slide = create_content_slide(prs, "Les 4P - Produit (Product)")
    # Add image
    add_image_to_slide(slide, f"{images_dir}/marketing_mix.png", 6.5, 2.5, 3, 2.25)
    left = Inches(0.5)
    top = Inches(2)
    width = Inches(5.5)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Offre de produit:",
        "  - Plateforme web et application mobile",
        "  - Gestion de budget et devis automatisés",
        "  - Catalogue de prestataires (traiteurs, DJ, photographes...)",
        "  - Outils de planification et timeline",
        "  - Liste d'invités et gestion des RSVP",
        "  - Templates personnalisables",
        "Fonctionnalités premium:",
        "  - Coordination avec prestataires en temps réel",
        "  - Analytics et rapports détaillés",
        "  - Support prioritaire 24/7"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 7: Les 4P - Prix
    slide = create_content_slide(prs, "Les 4P - Prix (Price)")
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Stratégie tarifaire: Freemium + Abonnement",
        "Version Gratuite:",
        "  - Fonctionnalités de base limitées",
        "  - 1 événement actif maximum",
        "  - Support communautaire",
        "Version Premium: 29€/mois ou 290€/an",
        "  - Événements illimités",
        "  - Toutes les fonctionnalités",
        "  - Support prioritaire",
        "Version Professionnelle: 99€/mois",
        "  - Multi-utilisateurs",
        "  - API et intégrations avancées",
        "  - Compte manager dédié"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 8: Les 4P - Place (Distribution)
    slide = create_content_slide(prs, "Les 4P - Place (Distribution)")
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Canaux de distribution:",
        "  - Site web principal (plateforme en ligne)",
        "  - Application mobile iOS et Android",
        "  - Stores: App Store et Google Play",
        "Accessibilité:",
        "  - 100% digital, accessible 24/7",
        "  - Compatible desktop, tablette, mobile",
        "  - Interface multilingue",
        "Points de contact:",
        "  - Réseaux sociaux (Instagram, Facebook, LinkedIn)",
        "  - Partenariats avec salles de fêtes et prestataires",
        "  - Présence sur événements professionnels (salons du mariage, etc.)"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 9: Les 4P - Promotion
    slide = create_content_slide(prs, "Les 4P - Promotion")
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Stratégie de communication:",
        "Marketing digital:",
        "  - SEO/SEM (référencement naturel et payant)",
        "  - Publicités Facebook et Instagram",
        "  - Content marketing (blog, guides pratiques)",
        "  - Email marketing et newsletters",
        "Marketing d'influence:",
        "  - Partenariats avec influenceurs lifestyle",
        "  - Témoignages clients et études de cas",
        "Promotions:",
        "  - 1 mois gratuit pour les nouveaux utilisateurs",
        "  - Programme de parrainage (réductions)",
        "  - Offres spéciales pour événements de grande envergure"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 10: Buyer Persona - Profil 1
    slide = create_content_slide(prs, "Buyer Persona - Thomas, le Futur Marié")
    # Add persona image on the left
    add_image_to_slide(slide, f"{images_dir}/persona_thomas.png", 0.5, 2, 2.8, 2.8)
    # Add text on the right
    left = Inches(3.5)
    top = Inches(1.8)
    width = Inches(6)
    height = Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Démographie:",
        "  - 30 ans, fiancé, va se marier dans 6 mois",
        "  - Ingénieur dans une entreprise tech",
        "  - Habite en zone urbaine, revenus confortables",
        "Objectifs:",
        "  - Organiser son propre mariage",
        "  - Créer un événement inoubliable pour lui et sa fiancée",
        "  - Optimiser le budget et gagner du temps",
        "  - Impressionner famille et amis",
        "Défis:",
        "  - Première expérience d'organisation de mariage",
        "  - Manque de temps avec son travail",
        "  - Difficulté à coordonner plusieurs prestataires",
        "  - Stress de la gestion du budget et des choix",
        "Comportement:",
        "  - Recherche en ligne de solutions et comparaisons",
        "  - Consulte forums et avis clients",
        "  - Valorise l'efficacité et la technologie"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 11: Buyer Persona - Profil 2
    slide = create_content_slide(prs, "Buyer Persona - Marc, le Pro de l'Événementiel")
    # Add persona image on the left
    add_image_to_slide(slide, f"{images_dir}/persona_marc.png", 0.5, 2, 2.8, 2.8)
    # Add text on the right
    left = Inches(3.5)
    top = Inches(1.8)
    width = Inches(6)
    height = Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Démographie:",
        "  - 38 ans, entrepreneur",
        "  - Directeur d'une agence événementielle",
        "  - Gère 20-30 événements par an",
        "Objectifs:",
        "  - Optimiser la gestion de ses événements",
        "  - Améliorer la satisfaction client",
        "  - Augmenter sa rentabilité",
        "Défis:",
        "  - Gestion de multiples projets simultanés",
        "  - Coordination d'équipes et prestataires",
        "  - Suivi financier complexe",
        "Comportement:",
        "  - Recherche des outils professionnels",
        "  - Valorise l'efficacité et le ROI",
        "  - Besoin de formations et support"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 12: Parcours d'achat - Étape 1 & 2
    slide = create_content_slide(prs, "Parcours d'Achat Idéal - Prise de conscience & Considération")
    # Add image at top
    add_image_to_slide(slide, f"{images_dir}/customer_journey.png", 3, 1.5, 4, 1.5)
    left = Inches(0.5)
    top = Inches(3.2)
    width = Inches(9)
    height = Inches(3.8)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "1. Prise de conscience (Awareness):",
        "  - Déclencheur: Événement à organiser",
        "  - Problème identifié: Complexité de l'organisation",
        "  - Recherche Google: 'comment organiser une fête', 'plateforme événement'",
        "  - Découverte via: Publicité, recommandation, article de blog",
        "  - Actions marketing: SEO, contenu éducatif, publicités ciblées",
        "",
        "2. Considération:",
        "  - Visite du site web",
        "  - Exploration des fonctionnalités",
        "  - Comparaison avec concurrents",
        "  - Lecture d'avis et témoignages",
        "  - Actions marketing: Démos gratuites, guides comparatifs, webinaires"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 13: Parcours d'achat - Étape 3 & 4
    slide = create_content_slide(prs, "Parcours d'Achat Idéal - Décision & Fidélisation")
    left = Inches(1)
    top = Inches(1.8)
    width = Inches(8)
    height = Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "3. Décision:",
        "  - Inscription gratuite ou essai",
        "  - Création du premier événement",
        "  - Expérience positive avec version gratuite",
        "  - Besoin de fonctionnalités avancées",
        "  - Décision d'upgrade vers version payante",
        "  - Actions marketing: Essai gratuit, onboarding personnalisé, promo launch",
        "",
        "4. Fidélisation et Advocacy:",
        "  - Utilisation régulière de la plateforme",
        "  - Événement réussi grâce à l'outil",
        "  - Renouvellement de l'abonnement",
        "  - Recommandation à l'entourage",
        "  - Actions marketing: Programme fidélité, support excellent, communauté"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 14: Parcours d'achat - Vue d'ensemble visuelle
    slide = create_content_slide(prs, "Parcours d'Achat - Vue Complète")
    left = Inches(0.5)
    top = Inches(2)
    width = Inches(9)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "AWARENESS → CONSIDÉRATION → DÉCISION → FIDÉLISATION",
        "",
        "Touchpoints clés:",
        "  • Google Ads / SEO → Site web → Inscription gratuite → Usage → Upgrade",
        "  • Réseaux sociaux → Landing page → Démo → Essai → Abonnement",
        "  • Recommandation → Visite → Comparaison → Achat → Parrainage",
        "",
        "Durée moyenne du cycle:",
        "  • Particuliers: 2-4 semaines",
        "  • Professionnels: 1-2 mois",
        "",
        "KPIs de suivi:",
        "  • Taux de conversion visiteur → inscrit: 15%",
        "  • Taux de conversion gratuit → payant: 8%",
        "  • Taux de rétention à 12 mois: 70%"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 15: Conclusion et prochaines étapes
    slide = create_content_slide(prs, "Conclusion et Prochaines Étapes")
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Résumé du plan marketing:",
        "  - Positionnement clair sur le marché de l'événementiel",
        "  - Mix marketing cohérent (4P) aligné sur nos cibles",
        "  - Connaissance approfondie de nos buyer personas",
        "  - Parcours d'achat optimisé pour la conversion",
        "",
        "Prochaines étapes:",
        "  1. Lancement de la campagne digitale (Mois 1-2)",
        "  2. Développement de partenariats stratégiques (Mois 2-3)",
        "  3. Amélioration continue basée sur les retours clients",
        "  4. Expansion géographique (Phase 2)",
        "",
        "Objectif année 1: 5000 utilisateurs actifs, 500 abonnés premium"
    ]
    add_bullet_points(textbox, points)
    
    # Save the presentation
    filename = "Plan_Marketing_Plateforme_Gestion_Fetes.pptx"
    prs.save(filename)
    print(f"Présentation créée avec succès: {filename}")
    return filename

if __name__ == "__main__":
    create_marketing_presentation()
