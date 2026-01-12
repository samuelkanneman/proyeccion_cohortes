# Tablero de Proyección de Mora - Chain Ladder

Aplicación interactiva con Streamlit para proyectar mora futura de cohortes utilizando metodología Chain Ladder.

## 📋 Requisitos

- Python 3.8 o superior
- Pip (gestor de paquetes de Python)

## 🚀 Instalación

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación:**
   ```bash
   streamlit run app_proyeccion_cohortes.py
   ```

3. La aplicación se abrirá automáticamente en tu navegador (generalmente en `http://localhost:8501`)

## 📊 Uso

### 1. Preparar el archivo CSV

El archivo debe tener la siguiente estructura:
- **Separador:** punto y coma (;)
- **Primera columna:** cohortes en formato YYYY-MM
- **Primera fila:** períodos en formato YYYY-MM
- **Valores:** mora >90d en formato porcentaje (ejemplo: 5,2% o 5.2%)

Ejemplo:
```csv
;2023-01;2023-02;2023-03;2023-04
2023-01;5,2%;8,1%;10,5%;12,3%
2023-02;;6,3%;9,2%;11,5%
2023-03;;;7,1%;10,2%
```

### 2. Cargar y proyectar

1. **Cargar archivo:** Usa el botón "Cargar archivo CSV" en la barra lateral
2. **Seleccionar cohorte:** Elige la cohorte que deseas proyectar
3. **Definir MOB objetivo:** Desliza el slider para seleccionar hasta qué MOB proyectar
4. **Proyectar:** Haz click en el botón "🚀 Proyectar"

### 3. Explorar resultados

La aplicación ofrece 4 pestañas:

- **📊 Visualizaciones:** Gráficos interactivos de la proyección vs histórico
- **📋 Tabla Detallada:** Datos observados y proyectados con intervalos de confianza
- **📈 Factores de Desarrollo:** Análisis de los factores históricos utilizados
- **💾 Exportar:** Descarga los resultados en CSV o Excel

## 🎯 Características

- **Proyección basada en Chain Ladder:** Utiliza factores de desarrollo históricos
- **Intervalos de confianza:** Calcula rangos basados en desviación estándar histórica
- **Visualización interactiva:** Gráficos dinámicos con Plotly
- **Comparación histórica:** Muestra el comportamiento de todas las cohortes
- **Exportación flexible:** Descarga resultados en CSV o Excel

## 📖 Metodología

La aplicación utiliza **Chain Ladder** para proyectar mora futura:

1. **Cálculo de factores:** Analiza cómo evolucionó la mora entre MOBs consecutivos en cohortes históricas
2. **Promedio histórico:** Calcula factores promedio con su variabilidad
3. **Proyección:** Aplica estos factores iterativamente a la cohorte objetivo

**Ejemplo:** Si históricamente la mora pasó de 10% en MOB 5 a 13% en MOB 6 (factor 1.3), se aplica ese factor a la cohorte proyectada.

## 🛠️ Troubleshooting

**Error de encoding al cargar CSV:**
- Asegúrate de que el archivo esté en UTF-8
- Verifica que el separador sea punto y coma (;)

**No aparecen cohortes para proyectar:**
- Verifica el formato de fechas (YYYY-MM)
- Confirma que hay datos válidos en el archivo

**Proyección no disponible hasta MOB deseado:**
- La proyección está limitada por los factores históricos disponibles
- Si quieres proyectar hasta MOB 18, necesitas cohortes que hayan llegado al menos a MOB 18

## 📝 Notas

- Los valores con formato español (5,2%) se parsean automáticamente
- Los MOBs sin datos históricos no pueden proyectarse
- El cálculo de intervalos usa ±1 desviación estándar

## 🔄 Actualizaciones Futuras Potenciales

- Selección de cohortes de referencia para factores
- Ajuste manual de factores
- Proyección de múltiples cohortes simultáneas
- Análisis de sensibilidad

---

**Desarrollado por:** Kanneman, Samuel 
**Versión:** 1.0  
**Fecha:** Enero 2026
