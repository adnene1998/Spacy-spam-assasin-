#!/usr/bin/env python3
"""
Script to create a marketing plan presentation for an event management platform.
Includes: SCP, 4Ps, Buyer Persona, and Ideal Buyer Journey.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

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
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
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
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4.5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Segment 1: Particuliers (mariages, anniversaires, fêtes familiales)",
        "  - Âge: 25-55 ans",
        "  - Revenus moyens à élevés",
        "  - Recherchent simplicité et qualité",
        "Segment 2: Professionnels de l'événementiel",
        "  - Organisateurs d'événements",
        "  - Entreprises (événements corporatifs)",
        "  - Besoin d'outils professionnels et de gestion",
        "Segment 3: Petites entreprises et associations",
        "  - Budget limité",
        "  - Événements récurrents"
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
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
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
    slide = create_content_slide(prs, "Buyer Persona - Sarah, l'Organisatrice Particulière")
    left = Inches(1)
    top = Inches(1.8)
    width = Inches(8)
    height = Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    points = [
        "Démographie:",
        "  - 32 ans, mariée, 1 enfant",
        "  - Cadre dans une entreprise de marketing",
        "  - Habite en zone urbaine, revenus confortables",
        "Objectifs:",
        "  - Organiser le mariage de sa sœur",
        "  - Créer un événement mémorable dans son budget",
        "  - Gagner du temps dans l'organisation",
        "Défis:",
        "  - Manque de temps avec son travail",
        "  - Difficulté à coordonner plusieurs prestataires",
        "  - Stress de la gestion du budget",
        "Comportement:",
        "  - Recherche en ligne de solutions",
        "  - Active sur les réseaux sociaux",
        "  - Lit les avis avant de choisir"
    ]
    add_bullet_points(textbox, points)
    
    # Slide 11: Buyer Persona - Profil 2
    slide = create_content_slide(prs, "Buyer Persona - Marc, le Pro de l'Événementiel")
    left = Inches(1)
    top = Inches(1.8)
    width = Inches(8)
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
    left = Inches(1)
    top = Inches(1.8)
    width = Inches(8)
    height = Inches(5)
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
