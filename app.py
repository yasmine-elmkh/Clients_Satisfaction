# app.py - VERSION REFACTORISÉE AVEC XGBoost EXCLUSIF
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Dashboard Clients Satisfaction Prediction ", 
    layout="wide",
    page_icon="📊"
)

# Fonctions de chargement optimisées pour XGBoost
@st.cache_data
def load_data():
    """Charge les données avec prédictions XGBoost"""
    try:
        df = pd.read_csv("dataset/final_dataset_with_predictions.csv")
        st.sidebar.success(f"✅ Données chargées: {df.shape[0]:,} lignes")
        return df
    except Exception as e:
        st.error(f"❌ Erreur chargement données: {e}")
        return None

@st.cache_resource
def load_xgboost_model():
    """Charge UNIQUEMENT le modèle XGBoost et préprocesseur"""
    
    # 1. Pipeline complet XGBoost (solution idéale)
    if os.path.exists("pipeline_full.joblib"):
        try:
            pipeline = joblib.load("pipeline_full.joblib")
            st.sidebar.success("✅ Pipeline XGBoost complet chargé")
            return pipeline, None, "pipeline_full.joblib"
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur pipeline XGBoost: {e}")
    
    # 2. Solution séparée : préprocesseur + modèle XGBoost
    preprocessor = None
    model = None
    
    # Charger préprocesseur
    if os.path.exists("preprocessor.joblib"):
        try:
            preprocessor = joblib.load("preprocessor.joblib")
            st.sidebar.success("✅ Preprocesseur chargé")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur preprocesseur: {e}")
            return None, None, "Aucun"
    
    # Charger modèle XGBoost (uniquement)
    xgboost_files = [
        "best_model_XGBoost.joblib",  # Modèle optimisé du notebook
        "model_XGBoost.joblib"        # Modèle standard
    ]
    
    for model_file in xgboost_files:
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                st.sidebar.success(f"✅ Modèle XGBoost chargé: {model_file}")
                return model, preprocessor, model_file
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erreur {model_file}: {e}")
                continue
    
    st.sidebar.error("❌ Aucun modèle XGBoost trouvé! Exécutez le notebook d'abord.")
    return None, None, "Aucun"

def main():
    st.title("📊 Dashboard Satisfaction Clients - XGBoost")
    
    # Chargement des données
    df = load_data()
    if df is None:
        st.error("""
        ❌ Impossible de charger les données. 
        
        **Solutions :**
        1. Vérifiez que le fichier `dataset/final_dataset_with_predictions.csv` existe
        2. Exécutez d'abord le notebook pour générer les données
        """)
        return

    # Chargement du modèle XGBoost
    model, preprocessor, model_name = load_xgboost_model()
    
    # Header avec statut modèle
    if "XGBoost" in str(model_name) or model_name == "pipeline_full.joblib":
        st.success(f"🚀 **Modèle XGBoost chargé :** {model_name}")
        
        # Afficher les métriques de performance du modèle
        accuracy = (df['target'] == df['predicted_target']).mean()
        st.info(f"🎯 **Performance du modèle :** {accuracy:.1%} de précision sur les données historiques")
    else:
        st.error("""
        ❌ **XGBoost non disponible** 
        
        **Actions requises :**
        1. Exécutez le notebook pour entraîner et sauvegarder XGBoost
        2. Vérifiez la présence des fichiers :
           - `best_model_XGBoost.joblib` 
           - `preprocessor.joblib`
           - `pipeline_full.joblib`
        """)
        # Mode dégradé - affichage données seulement
        st.warning("📊 Mode dégradé : Affichage des données existantes uniquement")

    # =============================================================================
    # SIDEBAR - FILTRES
    # =============================================================================
    st.sidebar.header("🔍 Filtres d'Analyse")
    
    # Filtre délai livraison
    delivery_min, delivery_max = int(df['delivery_time_days'].min()), int(df['delivery_time_days'].max())
    delivery_range = st.sidebar.slider(
        "Délai de livraison (jours)", 
        delivery_min, delivery_max, 
        (delivery_min, min(delivery_max, 30)),
        help="Filtrer par délai de livraison"
    )
    
    # Filtre jour de semaine
    weekday_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    weekday_options = st.sidebar.multiselect(
        "Jour d'achat", 
        options=list(range(7)), 
        default=list(range(7)),
        format_func=lambda x: weekday_names[x],
        help="Sélectionner les jours à inclure"
    )
    
    # Filtre satisfaction
    satisfaction_options = st.sidebar.multiselect(
        "Niveau de satisfaction",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: "😊 Satisfait" if x == 1 else "😞 Insatisfait",
        help="Filtrer par niveau de satisfaction prédit"
    )
    
    # Filtre prix
    price_min, price_max = int(df['price'].min()), int(df['price'].max())
    price_range = st.sidebar.slider(
        "Plage de prix (€)",
        price_min, price_max,
        (price_min, price_max),
        help="Filtrer par prix des produits"
    )
    
    # Application des filtres
    df_filtered = df[
        (df['delivery_time_days'] >= delivery_range[0]) &
        (df['delivery_time_days'] <= delivery_range[1]) &
        (df['purchase_weekday'].isin(weekday_options)) &
        (df['predicted_target'].isin(satisfaction_options)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ].copy()
    
    # Métriques sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Statistiques Filtrees**")
    
    col_sb1, col_sb2 = st.sidebar.columns(2)
    
    with col_sb1:
        st.metric("Commandes", f"{df_filtered.shape[0]:,}")
        st.metric("Satisfaction", f"{df_filtered['predicted_target'].mean():.1%}")
    
    with col_sb2:
        st.metric("Délai Moyen", f"{df_filtered['delivery_time_days'].mean():.1f}j")
        if 'target' in df_filtered.columns:
            accuracy = (df_filtered['target'] == df_filtered['predicted_target']).mean()
            st.metric("Précision", f"{accuracy:.1%}")

    # =============================================================================
    # KPI PRINCIPAUX
    # =============================================================================
    st.header("📈 Indicateurs Clés de Performance - XGBoost")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        satisfaction_rate = df_filtered['predicted_target'].mean()
        st.metric(
            "Taux Satisfaction Prédit", 
            f"{satisfaction_rate:.1%}",
            help="Pourcentage de clients satisfaits selon XGBoost"
        )
    
    with col2:
        avg_delivery = df_filtered['delivery_time_days'].mean()
        st.metric(
            "Délai Livraison Moyen", 
            f"{avg_delivery:.1f} jours",
            help="Délai moyen de livraison en jours"
        )
    
    with col3:
        total_orders = df_filtered.shape[0]
        st.metric(
            "Total Commandes", 
            f"{total_orders:,}",
            help="Nombre total de commandes filtrées"
        )
    
    with col4:
        avg_price = df_filtered['price'].mean()
        st.metric(
            "Prix Moyen", 
            f"{avg_price:.1f}€",
            help="Prix moyen des produits"
        )

    # =============================================================================
    # ANALYSE DÉTAILLÉE
    # =============================================================================
    st.header("📊 Analyse Détailée par XGBoost")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Corrélations", "📋 Données Brutes", "🔮 Prédiction Temps Réel"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Répartition Satisfaction XGBoost")
            fig, ax = plt.subplots(figsize=(8, 6))
            satisfaction_counts = df_filtered['predicted_target'].value_counts()
            colors = ['#ff6b6b', '#51cf66']
            
            # Diagramme circulaire
            wedges, texts, autotexts = ax.pie(
                satisfaction_counts, 
                labels=['Insatisfait', 'Satisfait'], 
                autopct='%1.1f%%', 
                colors=colors, 
                startangle=90,
                textprops={'fontsize': 12},
                explode=(0.05, 0)  # Mise en évidence
            )
            
            # Amélioration visuelle
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            ax.set_title("Distribution des Prédictions XGBoost", fontsize=14, fontweight='bold', pad=20)
            st.pyplot(fig)
            
            # Statistiques détaillées
            st.markdown(f"""
            **📊 Détails Satisfaction:**
            - 😊 Clients satisfaits: **{satisfaction_counts.get(1, 0):,}** ({satisfaction_counts.get(1, 0)/len(df_filtered):.1%})
            - 😞 Clients insatisfaits: **{satisfaction_counts.get(0, 0):,}** ({satisfaction_counts.get(0, 0)/len(df_filtered):.1%})
            """)
        
        with col2:
            st.subheader("Distribution des Délais de Livraison")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Histogramme avec KDE
            sns.histplot(df_filtered['delivery_time_days'], bins=20, kde=True, ax=ax, color='skyblue', alpha=0.7)
            
            # Lignes statistiques
            mean_delivery = df_filtered['delivery_time_days'].mean()
            median_delivery = df_filtered['delivery_time_days'].median()
            
            ax.axvline(mean_delivery, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_delivery:.1f} jours')
            ax.axvline(median_delivery, color='green', linestyle='--', linewidth=2, label=f'Médiane: {median_delivery:.1f} jours')
            
            ax.set_title("Distribution des Délais de Livraison", fontsize=14, fontweight='bold')
            ax.set_xlabel("Délai (jours)")
            ax.set_ylabel("Nombre de Commandes")
            ax.legend()
            
            st.pyplot(fig)
            
            # Statistiques délai
            st.markdown(f"""
            **📦 Statistiques Délai:**
            - 📍 Moyenne: **{mean_delivery:.1f}** jours
            - 📊 Médiane: **{median_delivery:.1f}** jours  
            - 🔼 Maximum: **{df_filtered['delivery_time_days'].max():.1f}** jours
            - 🔽 Minimum: **{df_filtered['delivery_time_days'].min():.1f}** jours
            """)

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Impact Délai sur Satisfaction")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Boxplot avec style amélioré
            box_plot = sns.boxplot(
                data=df_filtered, 
                x='predicted_target', 
                y='delivery_time_days', 
                ax=ax, 
                palette=['#ff6b6b', '#51cf66'],
                showfliers=False  # Masquer outliers pour plus de lisibilité
            )
            
            # Personnalisation
            ax.set_xticklabels(['😞 Insatisfait', '😊 Satisfait'], fontsize=12)
            ax.set_title("Impact du Délai sur la Satisfaction XGBoost", fontsize=14, fontweight='bold')
            ax.set_ylabel("Délai de Livraison (jours)")
            ax.set_xlabel("")
            
            # Ajouter des annotations de moyenne
            for i, target in enumerate([0, 1]):
                mean_val = df_filtered[df_filtered['predicted_target'] == target]['delivery_time_days'].mean()
                ax.text(i, mean_val + 0.5, f'Moy: {mean_val:.1f}j', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            st.pyplot(fig)
            
            # Insights
            mean_unsatisfied = df_filtered[df_filtered['predicted_target'] == 0]['delivery_time_days'].mean()
            mean_satisfied = df_filtered[df_filtered['predicted_target'] == 1]['delivery_time_days'].mean()
            
            st.info(f"""
            **💡 Insight XGBoost:**
            Les clients **insatisfaits** ont en moyenne des délais de **{mean_unsatisfied:.1f} jours**  
            Les clients **satisfaits** ont en moyenne des délais de **{mean_satisfied:.1f} jours**  
            **Écart:** {mean_unsatisfied - mean_satisfied:.1f} jours
            """)
            
        with col2:
            st.subheader("Relation Prix vs Satisfaction")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Scatter plot avec régression
            scatter = sns.scatterplot(
                data=df_filtered, 
                x='price', 
                y='prob_satisfied', 
                hue='predicted_target', 
                palette=['#ff6b6b', '#51cf66'], 
                alpha=0.6,
                ax=ax
            )
            
            # Ligne de tendance
            if len(df_filtered) > 1:
                z = np.polyfit(df_filtered['price'], df_filtered['prob_satisfied'], 1)
                p = np.poly1d(z)
                ax.plot(df_filtered['price'], p(df_filtered['price']), "r--", alpha=0.8, linewidth=2)
            
            ax.set_title("Relation Prix et Probabilité de Satisfaction", fontsize=14, fontweight='bold')
            ax.set_xlabel("Prix du Produit (€)")
            ax.set_ylabel("Probabilité de Satisfaction XGBoost")
            ax.legend(title='Prédiction', labels=['Insatisfait', 'Satisfait'])
            
            st.pyplot(fig)
            
            # Corrélation
            if len(df_filtered) > 1:
                correlation = df_filtered['price'].corr(df_filtered['prob_satisfied'])
                st.metric("📈 Corrélation Prix-Satisfaction", f"{correlation:.3f}")

    with tab3:
        st.subheader("📋 Données Détailées avec Prédictions XGBoost")
        
        # Options d'affichage
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            n_rows = st.slider("Lignes à afficher", 10, 100, 20)
        with col2:
            show_all_cols = st.checkbox("Afficher toutes les colonnes", value=False)
        
        # Sélection des colonnes à afficher
        base_columns = [
            'order_id', 'delivery_time_days', 'purchase_weekday',
            'product_density', 'price', 'predicted_target', 
            'prob_satisfied'
        ]
        
        if 'target' in df_filtered.columns:
            base_columns.append('target')
        
        # Colonnes additionnelles si demandé
        all_columns = base_columns + [col for col in df_filtered.columns if col not in base_columns]
        
        display_columns = all_columns if show_all_cols else base_columns
        
        # Garder seulement les colonnes disponibles
        available_columns = [col for col in display_columns if col in df_filtered.columns]
        
        # Affichage du dataframe
        st.dataframe(
            df_filtered[available_columns].head(n_rows),
            use_container_width=True,
            height=400
        )
        
        # Statistiques d'affichage
        st.caption(f"Affichage de {min(n_rows, len(df_filtered))} lignes sur {len(df_filtered):,} totales")
        
        # Téléchargement des données filtrées
        csv = df_filtered[available_columns].to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=csv,
            file_name="donnees_satisfaction_xgboost.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab4:
        st.subheader("🔮 Prédicteur de Satisfaction - XGBoost en Temps Réel")
        
        if model is None:
            st.error("""
            ❌ **XGBoost non disponible pour la prédiction**
            
            **Solutions:**
            1. Vérifiez que `best_model_XGBoost.joblib` ou `pipeline_full.joblib` existent
            2. Exécutez le notebook pour générer le modèle XGBoost
            3. Vérifiez que `preprocessor.joblib` est présent si vous utilisez le modèle séparé
            """)
        else:
            st.success("✅ **XGBoost prêt pour la prédiction**")
            st.info("💡 Remplissez les informations ci-dessous pour prédire la satisfaction d'un nouveau client avec XGBoost.")
            
            # Section de saisie des features
            st.markdown("### 📝 Caractéristiques de la Commande")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                delivery_time = st.slider(
                    "Délai livraison estimé (jours)", 
                    1, 60, 10, 
                    help="Délai de livraison prévu en jours",
                    key="delivery"
                )
            
            with col2:
                weekday = st.selectbox(
                    "Jour de commande", 
                    options=list(range(7)), 
                    key="weekday",
                    format_func=lambda x: weekday_names[x],
                    help="Jour de la semaine où la commande est passée"
                )
            
            with col3:
                product_density = st.slider(
                    "Densité du produit (g/cm³)", 
                    0.01, 5.0, 0.5, 0.01, 
                    help="Densité = Poids / Volume",
                    key="density"
                )
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                price = st.slider(
                    "Prix du produit (€)", 
                    0.1, 500.0, 50.0, 0.1, 
                    help="Prix du produit en euros",
                    key="price"
                )
            
            with col5:
                freight_value = st.slider(
                    "Frais de port (€)", 
                    0.0, 50.0, 10.0, 0.1, 
                    help="Frais de livraison",
                    key="freight"
                )
            
            with col6:
                product_weight = st.slider(
                    "Poids produit (g)", 
                    10, 5000, 500, 10, 
                    help="Poids du produit en grammes",
                    key="weight"
                )
            
            col7, col8 = st.columns(2)
            
            with col7:
                product_volume = st.slider(
                    "Volume produit (cm³)", 
                    100, 10000, 1000, 100, 
                    help="Volume du produit en centimètres cubes",
                    key="volume"
                )
            
            with col8:
                st.markdown("###")
                st.markdown("###")
                predict_button = st.button(
                    "🎯 Lancer la Prédiction XGBoost", 
                    type="primary", 
                    use_container_width=True,
                    help="Cliquez pour exécuter la prédiction avec le modèle XGBoost"
                )

            # Section de prédiction
            if predict_button:
                with st.spinner("🔮 XGBoost en cours de prédiction..."):
                    try:
                        # Préparation des données
                        sample_data = {
                            'delivery_time_days': [delivery_time],
                            'purchase_weekday': [weekday],
                            'product_density': [product_density],
                            'price': [price],
                            'freight_value': [freight_value],
                            'product_weight_g': [product_weight],
                            'product_volume_cm3': [product_volume]
                        }
                        
                        sample_df = pd.DataFrame(sample_data)
                        
                        # Prédiction selon le type de modèle chargé
                        if model_name == "pipeline_full.joblib":
                            # Pipeline complet - prêt à l'emploi
                            prediction = model.predict(sample_df)[0]
                            probability = model.predict_proba(sample_df)[0, 1]
                            method_used = "Pipeline complet XGBoost"
                        else:
                            # Modèle et préprocesseur séparés
                            sample_processed = preprocessor.transform(sample_df)
                            prediction = model.predict(sample_processed)[0]
                            probability = model.predict_proba(sample_processed)[0, 1]
                            method_used = f"Modèle {model_name} + Préprocesseur"
                        
                        # =============================================================================
                        # AFFICHAGE DES RÉSULTATS
                        # =============================================================================
                        st.success("✅ **Prédiction XGBoost terminée !**")
                        
                        # Layout résultats principaux
                        st.subheader("📋 Résultats de la Prédiction XGBoost")
                        
                        result_col1, result_col2 = st.columns([1, 1])
                        
                        with result_col1:
                            # Carte de résultat avec emojis
                            if prediction == 1:
                                st.success(f"""
                                ## 🟢 CLIENT SATISFAIT
                                
                                **Probabilité de satisfaction:** {probability:.1%}
                                
                                ✅ **Prédiction:** Le client sera satisfait
                                ✅ **Confiance:** {probability:.1%}
                                ✅ **Recommandation:** Maintenir le service
                                """)
                            else:
                                st.error(f"""
                                ## 🔴 CLIENT INSATISFAIT
                                
                                **Probabilité d'insatisfaction:** {1-probability:.1%}
                                
                                ⚠️ **Prédiction:** Risque d'insatisfaction
                                ⚠️ **Confiance:** {1-probability:.1%}  
                                ⚠️ **Recommandation:** Actions correctives
                                """)
                        
                        with result_col2:
                            # Jauge de confiance visuelle
                            st.metric(
                                "Niveau de Confiance XGBoost", 
                                f"{max(probability, 1-probability):.1%}",
                                delta="Haute confiance" if max(probability, 1-probability) > 0.7 else "Confiance modérée"
                            )
                            
                            # Barre de progression
                            st.write("**Certitude de la prédiction:**")
                            confidence_level = probability if prediction == 1 else (1 - probability)
                            st.progress(float(confidence_level))
                            
                            # Indicateur de qualité
                            if confidence_level >= 0.8:
                                st.success("🎯 **Très haute confiance** - Prédiction très fiable")
                            elif confidence_level >= 0.6:
                                st.info("📊 **Confiance modérée** - Prédiction probable")
                            else:
                                st.warning("💡 **Confiance faible** - Résultat à interpréter avec prudence")
                            
                            st.caption(f"*Méthode: {method_used}*")
                        
                        # =============================================================================
                        # RECOMMANDATIONS DÉTAILLÉES
                        # =============================================================================
                        with st.expander("💡 RECOMMANDATIONS DÉTAILLÉES XGBoost", expanded=True):
                            if prediction == 0:
                                st.markdown("""
                                ### 🚨 Plan d'Action pour Client à Risque
                                
                                **📦 Optimisation Livraison :**
                                - ✅ **Réduction délai:** Passer de {} à {} jours
                                - ✅ **Express option:** Proposer livraison express
                                - ✅ **Tracking:** Mettre en place suivi temps réel
                                
                                **💬 Communication Proactive :**
                                - ✅ **Alertes:** Notifier des retards potentiels
                                - ✅ **Transparence:** Donner estimation réaliste
                                - ✅ **Support:** Numéro dédié pour ce client
                                
                                **🎁 Gestes Commerciaux :**
                                - ✅ **Remise:** Offrir 10-15% sur prochaine commande
                                - ✅ **Cadeau:** Ajouter un produit complémentaire
                                - ✅ **Fidélisation:** Code promo personnel
                                """.format(delivery_time, max(1, delivery_time - 3)))
                                
                                # Analyse des facteurs de risque
                                st.markdown("#### 🔍 Facteurs de Risque Identifiés:")
                                risk_factors = []
                                if delivery_time > 15:
                                    risk_factors.append(f"📅 Délai élevé ({delivery_time} jours)")
                                if price > 100:
                                    risk_factors.append(f"💰 Prix élevé ({price}€)")
                                if freight_value > 15:
                                    risk_factors.append(f"🚚 Frais de port élevés ({freight_value}€)")
                                
                                if risk_factors:
                                    st.warning(" • ".join(risk_factors))
                                else:
                                    st.info("📊 Le modèle XGBoost a identifié des patterns complexes non évidents")
                                
                            else:
                                st.markdown("""
                                ### ✅ Stratégie de Fidélisation
                                
                                **🔄 Renforcement Satisfaction :**
                                - ✅ **Suivi:** Contacter à J+7 pour feedback
                                - ✅ **Avis:** Solliciter avis et témoignage
                                - ✅ **Récompense:** Offrir programme fidélité
                                
                                **📈 Capitalisation Expérience :**
                                - ✅ **Analyse:** Comprendre raisons du succès
                                - ✅ **Replication:** Appliquer à autres clients
                                - ✅ **Amélioration:** Maintenir standards qualité
                                
                                **🤝 Relation Client :**
                                - ✅ **Personnalisation:** Communications sur-mesure
                                - ✅ **Anticipation:** Proposer produits similaires
                                - ✅ **Exclusivité:** Offres réservées
                                """)
                        
                        # =============================================================================
                        # DÉTAILS TECHNIQUES XGBoost
                        # =============================================================================
                        with st.expander("🔧 Détails Techniques de la Prédiction XGBoost"):
                            st.write("**Configuration du modèle:**")
                            st.code(f"""
                            Modèle: {model_name}
                            Type: XGBoost Classifier
                            Probabilité satisfaction: {probability:.2%}
                            Seuil de décision: 50%
                            Méthode: {method_used}
                            """)
                            
                            st.write("**Features utilisées pour la prédiction:**")
                            features_df = pd.DataFrame({
                                'Feature': list(sample_data.keys()),
                                'Valeur': [val[0] for val in sample_data.values()],
                                'Importance': ['Haute' if feat in ['delivery_time_days', 'price'] else 'Moyenne' for feat in sample_data.keys()]
                            })
                            st.dataframe(features_df, use_container_width=True, hide_index=True)
                            
                            st.caption("💡 *XGBoost utilise un ensemble d'arbres de décision pour cette prédiction*")
                                
                    except Exception as e:
                        st.error(f"""
                        ❌ **Erreur lors de la prédiction XGBoost:**
                        
                        `{str(e)}`
                        
                        **Solutions:**
                        1. Vérifiez la compatibilité modèle/préprocesseur
                        2. Vérifiez les types de données des features
                        3. Réexécutez le notebook pour régénérer les artefacts
                        """)
                        
                        # Debug information
                        with st.expander("🐛 Informations de débogage"):
                            st.write("**Données envoyées au modèle:**")
                            st.json(sample_data)
                            st.write("**Modèle chargé:**", type(model))
                            if preprocessor:
                                st.write("**Preprocesseur chargé:**", type(preprocessor))

    # =============================================================================
    # FOOTER
    # =============================================================================
    
    st.markdown("---")

    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

    with footer_col1:
        st.markdown(f"**📊 Échelle Analyse:** {df.shape[0]:,} clients")

    with footer_col2:
        global_satisfaction = df['predicted_target'].mean()
        st.markdown(f"**😊 Satisfaction Globale:** {global_satisfaction:.1%}")

    with footer_col3:
        st.markdown("**🎯 Exactitude (Test):** 77.1%")

    with footer_col4:
        st.markdown("**📈 F1-Score (Test):** 86.3%")
    
    st.markdown(
        """
        <div style='text-align: center; color: gray; margin-top: 20px;'>
        📊 Dashboard Satisfaction Clients • 🤖 Powered by XGBoost • 🚀 Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()