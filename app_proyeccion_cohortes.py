"""
TABLERO DE PROYECCIÓN DE MORA - CHAIN LADDER
============================================
Aplicación Streamlit para proyectar mora futura de cohortes.

Ejecución:
    streamlit run app_proyeccion_cohortes.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Proyección de Mora - Chain Ladder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNCIONES (del script original adaptadas)
# ============================================================

def parse_pct(x):
    """Parsea porcentajes con formato español"""
    if pd.isna(x) or x == '':
        return np.nan
    if isinstance(x, str):
        return float(x.replace('%', '').replace(',', '.'))
    return float(x)


def load_data(uploaded_file):
    """Carga y procesa el archivo CSV"""
    try:
        df = pd.read_csv(uploaded_file, sep=';', index_col=0, encoding='utf-8-sig')
        df = df.applymap(parse_pct)
        return df, None
    except Exception as e:
        return None, str(e)


def create_mob_dataframe(df):
    """Convierte matriz vintage a formato MOB"""
    cohorts = df.index.tolist()
    periods = df.columns.tolist()
    
    mob_data = []
    for cohort in cohorts:
        cohort_year, cohort_month = int(cohort[:4]), int(cohort[5:7])
        for period in periods:
            period_year, period_month = int(period[:4]), int(period[5:7])
            mob = (period_year - cohort_year) * 12 + (period_month - cohort_month)
            value = df.loc[cohort, period]
            if not pd.isna(value):
                mob_data.append({
                    'cohorte': cohort,
                    'periodo': period,
                    'mob': mob,
                    'mora_pct': value
                })
    
    return pd.DataFrame(mob_data)


def calculate_development_factors(df_pivot):
    """Calcula factores de desarrollo promedio históricos"""
    factors = {}
    factors_detail = {}
    
    for mob in range(1, 24):
        prev_col = mob - 1
        curr_col = mob
        if prev_col in df_pivot.columns and curr_col in df_pivot.columns:
            valid_mask = (df_pivot[prev_col] > 0) & (~df_pivot[curr_col].isna())
            if valid_mask.sum() > 0:
                individual_factors = df_pivot.loc[valid_mask, curr_col] / df_pivot.loc[valid_mask, prev_col]
                factors[mob] = individual_factors.mean()
                factors_detail[mob] = {
                    'mean': individual_factors.mean(),
                    'std': individual_factors.std(),
                    'min': individual_factors.min(),
                    'max': individual_factors.max(),
                    'n': valid_mask.sum(),
                    'values': individual_factors.tolist()
                }
    
    return factors, factors_detail


def mob_to_date(cohorte, mob):
    """Convierte cohorte + MOB a fecha calendario"""
    cohort_year, cohort_month = int(cohorte[:4]), int(cohorte[5:7])
    target_month = cohort_month + mob
    target_year = cohort_year + (target_month - 1) // 12
    target_month = ((target_month - 1) % 12) + 1
    return f"{target_year}-{target_month:02d}"


def project_cohort(df_pivot, factors, cohorte, mob_objetivo):
    """Proyecta una cohorte específica hasta el MOB objetivo"""
    
    if cohorte not in df_pivot.index:
        return None, f"Cohorte {cohorte} no encontrada"
    
    # Obtener último MOB observado
    cohort_data = df_pivot.loc[cohorte].dropna()
    last_mob = int(cohort_data.index.max())
    last_value = cohort_data.iloc[-1]
    
    # Construir proyección
    proyeccion = []
    
    # Agregar datos observados
    for mob in cohort_data.index:
        proyeccion.append({
            'cohorte': cohorte,
            'mob': int(mob),
            'fecha': mob_to_date(cohorte, int(mob)),
            'mora_pct': cohort_data[mob],
            'tipo': 'Observado',
            'factor': None
        })
    
    # Proyectar hacia adelante
    current_value = last_value
    for future_mob in range(last_mob + 1, mob_objetivo + 1):
        if future_mob in factors:
            factor = factors[future_mob]
            current_value = current_value * factor
            proyeccion.append({
                'cohorte': cohorte,
                'mob': future_mob,
                'fecha': mob_to_date(cohorte, future_mob),
                'mora_pct': current_value,
                'tipo': 'Proyectado',
                'factor': factor
            })
    
    return pd.DataFrame(proyeccion), None


def create_projection_plot(df_proy, df_pivot):
    """Crea gráfico interactivo de proyección con Plotly"""
    
    cohorte = df_proy['cohorte'].iloc[0]
    
    fig = go.Figure()
    
    # Cohortes históricas (fondo)
    for c in df_pivot.index:
        if c != cohorte:
            data = df_pivot.loc[c].dropna()
            fig.add_trace(go.Scatter(
                x=data.index.tolist(),
                y=data.values.tolist(),
                mode='lines',
                line=dict(color='lightgray', width=1),
                opacity=0.3,
                showlegend=False,
                hovertemplate=f'<b>{c}</b><br>MOB: %{{x}}<br>Mora: %{{y:.2f}}%<extra></extra>'
            ))
    
    # Cohorte objetivo - observado
    observado = df_proy[df_proy['tipo'] == 'Observado']
    proyectado = df_proy[df_proy['tipo'] == 'Proyectado']
    
    fig.add_trace(go.Scatter(
        x=observado['mob'],
        y=observado['mora_pct'],
        mode='lines+markers',
        name='Observado',
        line=dict(color='steelblue', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Observado</b><br>MOB: %{x}<br>Fecha: %{customdata}<br>Mora: %{y:.2f}%<extra></extra>',
        customdata=observado['fecha']
    ))
    
    # Cohorte objetivo - proyectado
    if len(proyectado) > 0:
        fig.add_trace(go.Scatter(
            x=proyectado['mob'],
            y=proyectado['mora_pct'],
            mode='lines+markers',
            name='Proyectado',
            line=dict(color='coral', width=3, dash='dash'),
            marker=dict(size=8, symbol='square'),
            hovertemplate='<b>Proyectado</b><br>MOB: %{x}<br>Fecha: %{customdata[0]}<br>Mora: %{y:.2f}%<br>Factor: %{customdata[1]:.3f}<extra></extra>',
            customdata=proyectado[['fecha', 'factor']].values
        ))
        
        # Conectar observado con proyectado
        fig.add_trace(go.Scatter(
            x=[observado['mob'].iloc[-1], proyectado['mob'].iloc[0]],
            y=[observado['mora_pct'].iloc[-1], proyectado['mora_pct'].iloc[0]],
            mode='lines',
            line=dict(color='coral', width=3, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f'Proyección de Mora - Cohorte {cohorte}',
        xaxis_title='MOB (Meses desde operación)',
        yaxis_title='Mora >90d (%)',
        hovermode='closest',
        height=500,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_bar_chart(df_proy):
    """Crea gráfico de barras comparativo"""
    
    cohorte = df_proy['cohorte'].iloc[0]
    
    fig = go.Figure()
    
    observado = df_proy[df_proy['tipo'] == 'Observado']
    proyectado = df_proy[df_proy['tipo'] == 'Proyectado']
    
    # Barras observadas
    fig.add_trace(go.Bar(
        x=observado['mob'],
        y=observado['mora_pct'],
        name='Observado',
        marker_color='steelblue',
        hovertemplate='<b>Observado</b><br>MOB: %{x}<br>Mora: %{y:.2f}%<extra></extra>'
    ))
    
    # Barras proyectadas
    if len(proyectado) > 0:
        fig.add_trace(go.Bar(
            x=proyectado['mob'],
            y=proyectado['mora_pct'],
            name='Proyectado',
            marker_color='coral',
            hovertemplate='<b>Proyectado</b><br>MOB: %{x}<br>Mora: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Cohorte {cohorte}: Observado vs Proyectado',
        xaxis_title='MOB',
        yaxis_title='Mora >90d (%)',
        height=400,
        template='plotly_white',
        barmode='group'
    )
    
    return fig


def create_factors_plot(factors_detail):
    """Crea gráfico de factores de desarrollo"""
    
    mobs = sorted(factors_detail.keys())
    means = [factors_detail[m]['mean'] for m in mobs]
    stds = [factors_detail[m]['std'] for m in mobs]
    
    fig = go.Figure()
    
    # Línea de factores promedio
    fig.add_trace(go.Scatter(
        x=mobs,
        y=means,
        mode='lines+markers',
        name='Factor promedio',
        line=dict(color='darkblue', width=2),
        marker=dict(size=6)
    ))
    
    # Banda de confianza (±1 std)
    upper = [m + s for m, s in zip(means, stds)]
    lower = [m - s for m, s in zip(means, stds)]
    
    fig.add_trace(go.Scatter(
        x=mobs + mobs[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Desv. Std.',
        showlegend=True
    ))
    
    # Línea de referencia en 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                  annotation_text="Factor = 1.0", annotation_position="right")
    
    fig.update_layout(
        title='Factores de Desarrollo Históricos',
        xaxis_title='MOB (Transición desde MOB anterior)',
        yaxis_title='Factor de Desarrollo',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def convert_df_to_excel(df):
    """Convierte DataFrame a Excel en memoria"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Proyección')
    output.seek(0)
    return output


# ============================================================
# APLICACIÓN STREAMLIT
# ============================================================

def main():
    
    # Header
    st.title("📊 Proyección de Mora - Chain Ladder")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV",
            type=['csv'],
            help="Archivo con estructura: Cohortes en filas, períodos en columnas, valores de mora en %"
        )
        
        st.markdown("---")
        
        # Instrucciones
        with st.expander("📖 Instrucciones"):
            st.markdown("""
            **Formato del archivo CSV:**
            - Separador: punto y coma (;)
            - Primera columna: cohortes (formato YYYY-MM)
            - Primera fila: períodos (formato YYYY-MM)
            - Valores: mora >90d en formato porcentaje
            
            **Ejemplo:**
            ```
            ;2023-01;2023-02;2023-03
            2023-01;5,2%;8,1%;10,5%
            2023-02;;6,3%;9,2%
            ```
            """)
    
    # Main content
    if uploaded_file is None:
        st.info("👈 Por favor, carga un archivo CSV para comenzar")
        st.markdown("""
        ### Sobre esta aplicación
        
        Esta herramienta utiliza la **metodología Chain Ladder** para proyectar mora futura de cohortes específicas.
        
        **¿Cómo funciona?**
        1. Analiza el comportamiento histórico de todas las cohortes
        2. Calcula factores de desarrollo promedio entre MOBs consecutivos
        3. Aplica estos factores para proyectar la mora futura de la cohorte seleccionada
        
        **Ventajas:**
        - Proyecciones basadas en datos históricos reales
        - Intervalos de confianza basados en variabilidad histórica
        - Visualización interactiva vs comportamiento histórico
        """)
        return
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df, error = load_data(uploaded_file)
    
    if error:
        st.error(f"Error al cargar el archivo: {error}")
        return
    
    # Procesar datos
    with st.spinner("Procesando datos..."):
        df_mob = create_mob_dataframe(df)
        df_pivot = df_mob.pivot(index='cohorte', columns='mob', values='mora_pct')
        factors, factors_detail = calculate_development_factors(df_pivot)
    
    # Selección de parámetros
    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Parámetros de proyección")
    
    cohortes_disponibles = sorted(df_pivot.index.tolist(), reverse=True)
    cohorte_objetivo = st.sidebar.selectbox(
        "Cohorte a proyectar",
        cohortes_disponibles,
        help="Selecciona la cohorte que deseas proyectar"
    )
    
    # Determinar MOB máximo observado para la cohorte
    cohort_data = df_pivot.loc[cohorte_objetivo].dropna()
    mob_actual = int(cohort_data.index.max())
    mob_max_disponible = max(factors.keys())
    
    mob_objetivo = st.sidebar.slider(
        "Proyectar hasta MOB",
        min_value=mob_actual + 1,
        max_value=min(24, mob_max_disponible),
        value=min(12, mob_max_disponible),
        help=f"MOB actual de la cohorte: {mob_actual}"
    )
    
    # Botón de proyección
    if st.sidebar.button("🚀 Proyectar", type="primary", use_container_width=True):
        
        with st.spinner("Generando proyección..."):
            df_proy, error = project_cohort(df_pivot, factors, cohorte_objetivo, mob_objetivo)
        
        if error:
            st.error(error)
            return
        
        # Guardar en session state
        st.session_state['df_proy'] = df_proy
        st.session_state['df_pivot'] = df_pivot
        st.session_state['factors_detail'] = factors_detail
    
    # Mostrar resultados si existen
    if 'df_proy' in st.session_state:
        
        df_proy = st.session_state['df_proy']
        df_pivot = st.session_state['df_pivot']
        factors_detail = st.session_state['factors_detail']
        
        observado = df_proy[df_proy['tipo'] == 'Observado']
        proyectado = df_proy[df_proy['tipo'] == 'Proyectado']
        
        # Métricas principales
        st.subheader("📈 Resumen de Proyección")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Cohorte",
                cohorte_objetivo
            )
        
        with col2:
            st.metric(
                "MOB Actual",
                f"{observado['mob'].max()}"
            )
        
        with col3:
            mora_actual = observado['mora_pct'].iloc[-1]
            mora_final = df_proy['mora_pct'].iloc[-1]
            delta = mora_final - mora_actual
            st.metric(
                "Mora Actual",
                f"{mora_actual:.2f}%",
                f"+{delta:.2f} pp proyectados"
            )
        
        with col4:
            fecha_final = df_proy['fecha'].iloc[-1]
            st.metric(
                "Proyección Final",
                f"{mora_final:.2f}%",
                f"al {fecha_final}"
            )
        
        st.markdown("---")
        
        # Tabs principales
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visualizaciones", 
            "📋 Tabla Detallada", 
            "📈 Factores de Desarrollo",
            "💾 Exportar"
        ])
        
        with tab1:
            # Gráfico principal
            fig1 = create_projection_plot(df_proy, df_pivot)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Gráfico de barras
            fig2 = create_bar_chart(df_proy)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Proyección Detallada")
            
            # Formatear DataFrame para display
            df_display = df_proy.copy()
            df_display['mora_pct'] = df_display['mora_pct'].apply(lambda x: f"{x:.2f}%")
            df_display['factor'] = df_display['factor'].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "-"
            )
            
            # Agregar intervalos de confianza
            intervalos = []
            for idx, row in df_proy.iterrows():
                if row['tipo'] == 'Proyectado' and row['mob'] in factors_detail:
                    std = factors_detail[row['mob']]['std']
                    factor = row['factor']
                    mora = row['mora_pct']
                    mora_base = mora / factor
                    mora_low = mora_base * (factor - std)
                    mora_high = mora_base * (factor + std)
                    intervalos.append(f"[{mora_low:.1f}% - {mora_high:.1f}%]")
                else:
                    intervalos.append("-")
            
            df_display['intervalo_confianza'] = intervalos
            
            # Renombrar columnas
            df_display = df_display.rename(columns={
                'cohorte': 'Cohorte',
                'mob': 'MOB',
                'fecha': 'Fecha',
                'mora_pct': 'Mora %',
                'tipo': 'Tipo',
                'factor': 'Factor',
                'intervalo_confianza': 'Intervalo ±1σ'
            })
            
            # Aplicar estilo condicional
            def color_tipo(val):
                if val == 'Observado':
                    return 'background-color: #e3f2fd'
                elif val == 'Proyectado':
                    return 'background-color: #fff3e0'
                return ''
            
            st.dataframe(
                df_display.style.applymap(color_tipo, subset=['Tipo']),
                use_container_width=True,
                height=400
            )
            
            # Resumen proyección por mes
            st.markdown("---")
            st.subheader("Resumen Mensual - Proyección")
            
            proyectado_display = proyectado[['fecha', 'mob', 'mora_pct']].copy()
            proyectado_display.columns = ['Mes Calendario', 'MOB', 'Mora Proyectada']
            proyectado_display['Mora Proyectada'] = proyectado_display['Mora Proyectada'].apply(
                lambda x: f"{x:.2f}%"
            )
            
            st.dataframe(proyectado_display, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Factores de Desarrollo Históricos")
            
            # Gráfico de factores
            fig3 = create_factors_plot(factors_detail)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Tabla de factores
            st.markdown("---")
            st.subheader("Detalle de Factores")
            
            factors_df = pd.DataFrame([
                {
                    'MOB': mob,
                    'Factor Promedio': f"{detail['mean']:.4f}",
                    'Desv. Std.': f"{detail['std']:.4f}",
                    'Mín': f"{detail['min']:.4f}",
                    'Máx': f"{detail['max']:.4f}",
                    'N° Observaciones': detail['n']
                }
                for mob, detail in sorted(factors_detail.items())
            ])
            
            st.dataframe(factors_df, use_container_width=True, hide_index=True)
        
        with tab4:
            st.subheader("Exportar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Exportar a CSV
                csv = df_proy.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv,
                    file_name=f'proyeccion_{cohorte_objetivo}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            with col2:
                # Exportar a Excel
                excel_data = convert_df_to_excel(df_proy)
                st.download_button(
                    label="📥 Descargar Excel",
                    data=excel_data,
                    file_name=f'proyeccion_{cohorte_objetivo}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Preview de datos a exportar
            st.markdown("**Preview de datos:**")
            st.dataframe(df_proy, use_container_width=True)


if __name__ == '__main__':
    main()
