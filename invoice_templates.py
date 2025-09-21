"""
Templates y prompts específicos para diferentes tipos de facturas
Extrae: CAE, importes, CUIT/RAZON SOCIAL
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class InvoiceTemplate:
    """Template para un proveedor específico por CUIT"""
    cuit: str  # CUIT del proveedor
    provider_name: str  # Nombre del proveedor
    prompt: str  # Prompt específico para este proveedor
    json_schema: Dict[str, Any]  # Esquema JSON específico

# Templates específicos por CUIT
PROVIDER_TEMPLATES = {
    "30685889397": InvoiceTemplate(
        cuit="30685889397",
        provider_name="TELECOM ARGENTINA S.A.",
        prompt="""
        Analiza esta factura de TELECOM ARGENTINA S.A. y extrae EXACTAMENTE la siguiente información en formato JSON:

        - CAE (Código de Autorización Electrónica)
        - CUIT del emisor
        - Razón Social completa del emisor
        - importe_total: El monto final total a pagar
        - saldo_anterior: Saldo del período anterior
        - total_factura: TOTAL CARGOS DEL MES (antes de sumar saldo anterior)
        - iva_21: Importe del IVA 21%
        - percep_iibb_bs_as: PERCEP. IIBB BS. AS. (Percepción Ingresos Brutos Buenos Aires)
        - subtotal: Subtotal antes de impuestos

        IMPORTANTE: Buscar específicamente "PERCEP. IIBB BS. AS." en el texto.

        Responde ÚNICAMENTE con un JSON válido sin explicaciones adicionales.
        """,
        json_schema={
            "cae": "",
            "cuit": "",
            "razon_social": "",
            "importe_total": "",
            "saldo_anterior": "",
            "total_factura": "",
            "iva_21": "",
            "percep_iibb_bs_as": "",
            "subtotal": "",
            "archivo_procesado": ""
        }
    ),

    "30663288497": InvoiceTemplate(
        cuit="30663288497",
        provider_name="AMX ARGENTINA S.A.",
        prompt="""
        Analiza esta factura de AMX ARGENTINA S.A. y extrae EXACTAMENTE la siguiente información en formato JSON:

        - CAE (Código de Autorización Electrónica)
        - CUIT del emisor
        - Razón Social completa del emisor
        - importe_total_impuestos: Total de todos los impuestos
        - importe_total_factura: Total general de la factura
        - iva_3: IVA 3% si existe
        - iva_21: IVA 21%
        - iva_27: IVA 27% si existe
        - impuesto_interno_5_2632: Impuesto interno 5% Ley 26.32 si existe
        - percep_iibb_bs_as: PERCEP. IIBB BS. AS. (Percepción Ingresos Brutos Buenos Aires)
        - total_factura: Total a pagar final

        IMPORTANTE: Buscar todos los tipos de IVA y impuestos específicos de AMX.

        Responde ÚNICAMENTE con un JSON válido sin explicaciones adicionales.
        """,
        json_schema={
            "cae": "",
            "cuit": "",
            "razon_social": "",
            "importe_total_impuestos": "",
            "importe_total_factura": "",
            "iva_3": "",
            "iva_21": "",
            "iva_27": "",
            "impuesto_interno_5_2632": "",
            "percep_iibb_bs_as": "",
            "total_factura": "",
            "archivo_procesado": ""
        }
    )
}

# Código legacy eliminado - ahora usamos PROVIDER_TEMPLATES específicos por CUIT

def detect_provider_by_cuit(text: str) -> str:
    """
    Detecta el proveedor basado en CUIT encontrado en el texto
    """
    text_upper = text.upper()

    # Imprimir texto para debug
    print(f"🔍 Texto para detección: {text_upper[:300]}...")

    # Buscar CUITs conocidos en el texto
    for cuit, template in PROVIDER_TEMPLATES.items():
        # Buscar el CUIT tal como está y también sin guiones
        cuit_variations = [
            cuit,  # Con guiones: 30-68588939-7
            cuit.replace("-", ""),  # Sin guiones: 30685889397
            cuit.replace("-", " "),  # Con espacios: 30 68588939 7
        ]

        for cuit_var in cuit_variations:
            if cuit_var in text_upper:
                print(f"✅ Proveedor detectado: {template.provider_name} (CUIT: {cuit})")
                return cuit

    print("❌ No se detectó CUIT conocido")
    return "UNKNOWN"

def get_template_for_cuit(cuit: str) -> InvoiceTemplate:
    """
    Obtiene el template para un CUIT específico
    """
    if cuit in PROVIDER_TEMPLATES:
        return PROVIDER_TEMPLATES[cuit]

    # Template genérico por defecto
    return InvoiceTemplate(
        cuit="UNKNOWN",
        provider_name="PROVEEDOR DESCONOCIDO",
        prompt="""
        Analiza esta factura y extrae la siguiente información básica en formato JSON:
        - CAE (Código de Autorización Electrónica)
        - CUIT del emisor
        - Razón Social completa del emisor
        - Importe total a pagar

        Responde ÚNICAMENTE con un JSON válido sin explicaciones adicionales.
        """,
        json_schema={
            "cae": "",
            "cuit": "",
            "razon_social": "",
            "importe_total": "",
            "archivo_procesado": ""
        }
    )

def create_enhanced_prompt_for_cuit(cuit: str) -> str:
    """
    Crea un prompt específico para el CUIT detectado
    """
    template = get_template_for_cuit(cuit)

    json_example = template.json_schema

    enhanced_prompt = f"""
{template.prompt}

El JSON debe tener exactamente esta estructura:
{json_example}

Instrucciones específicas:
- CAE: Buscar "CAE", "Código de Autorización", números de 14 dígitos
- CUIT: Formato XX-XXXXXXXX-X
- Razón Social: Nombre completo legal de la empresa
- Importes: Solo números, sin símbolo de moneda ni separadores de miles
- Si no encuentras un campo, usar cadena vacía ""

IMPORTANTE: Responder SOLO con JSON válido, sin texto adicional.
"""

    return enhanced_prompt