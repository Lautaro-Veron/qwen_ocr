"""
Procesador de facturas con Qwen2.5-VL
Extrae CAE, importes, CUIT/RAZON SOCIAL en formato JSON
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import re

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    print("⚠️ PyMuPDF no instalado. Para procesar PDFs, instalar con: pip install PyMuPDF")
    PDF_SUPPORT = False

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from invoice_templates import detect_provider_by_cuit, get_template_for_cuit, create_enhanced_prompt_for_cuit

class QwenInvoiceProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """Inicializa el procesador de facturas"""
        print("📥 Cargando modelo Qwen2.5-VL...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="float16",
            device_map="cuda"
        )

        print("📥 Cargando processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)

        print("✅ Modelo cargado exitosamente")

    def convert_pdf_to_image(self, pdf_path: str) -> str:
        """Convierte PDF a imagen temporal usando PyMuPDF"""
        if not PDF_SUPPORT:
            raise ImportError("PyMuPDF requerido para procesar PDFs")

        print(f"🔄 Convirtiendo PDF a imagen: {pdf_path}")

        try:
            # Abrir PDF con PyMuPDF
            doc = fitz.open(pdf_path)

            if len(doc) == 0:
                raise ValueError(f"PDF vacío: {pdf_path}")

            # Convertir primera página a imagen
            page = doc[0]  # Primera página

            # Renderizar página como imagen (matriz de transformación para alta resolución)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~200 DPI
            pix = page.get_pixmap(matrix=mat)

            # Guardar imagen temporal
            temp_image_path = pdf_path.replace('.pdf', '_temp.png')
            pix.save(temp_image_path)

            # Cerrar documento
            doc.close()

            print(f"✅ PDF convertido a: {temp_image_path}")
            return temp_image_path

        except Exception as e:
            print(f"❌ Error convirtiendo PDF: {e}")
            raise

    def cleanup_temp_image(self, temp_image_path: str):
        """Elimina imagen temporal"""
        try:
            if os.path.exists(temp_image_path) and '_temp.png' in temp_image_path:
                os.remove(temp_image_path)
                print(f"🗑️ Imagen temporal eliminada: {temp_image_path}")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar imagen temporal: {e}")

    def extract_text_preview(self, image_path: str) -> str:
        """Extrae preview del texto para detección de proveedor"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Extract the main company name and key text from this invoice"}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else ""

    def process_invoice(self, image_path: str) -> Dict[str, Any]:
        """
        Procesa una factura y extrae la información en formato JSON
        """
        print(f"📷 Procesando: {image_path}")

        # Manejar PDFs
        temp_image_path = None
        processing_path = image_path

        if image_path.lower().endswith('.pdf'):
            temp_image_path = self.convert_pdf_to_image(image_path)
            processing_path = temp_image_path

        try:
            # Paso 1: Detectar proveedor por CUIT
            print("🔍 Detectando proveedor por CUIT...")
            preview_text = self.extract_text_preview(processing_path)
            detected_cuit = detect_provider_by_cuit(preview_text)
            template = get_template_for_cuit(detected_cuit)

            print(f"🏢 Proveedor detectado: {template.provider_name} (CUIT: {detected_cuit})")

            # Paso 2: Crear prompt específico para el CUIT
            enhanced_prompt = create_enhanced_prompt_for_cuit(detected_cuit)

            # Paso 3: Procesar con prompt específico
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processing_path},
                        {"type": "text", "text": enhanced_prompt}
                    ]
                }
            ]

            print("🔧 Preparando datos para inferencia...")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")

            print("🚀 Extrayendo información...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Paso 4: Parsear JSON resultado
            raw_output = output_text[0] if output_text else ""

            try:
                # Intentar extraer JSON del output
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                else:
                    raise ValueError("No se encontró JSON válido en la respuesta")

                # Agregar metadatos
                result["cuit_detectado"] = detected_cuit
                result["proveedor_detectado"] = template.provider_name
                result["archivo_procesado"] = os.path.basename(image_path)

                return result

            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Error parseando JSON: {e}")
                print(f"Output crudo: {raw_output}")

                # Retornar estructura básica con el texto crudo
                return {
                    "error": f"Error parseando JSON: {str(e)}",
                    "raw_output": raw_output,
                    "cuit_detectado": detected_cuit,
                    "proveedor_detectado": template.provider_name,
                    "archivo_procesado": os.path.basename(image_path)
                }

        finally:
            # Limpiar imagen temporal si existe
            if temp_image_path:
                self.cleanup_temp_image(temp_image_path)

    def process_batch(self, images_directory: str, output_file: str = "invoice_results.json") -> List[Dict[str, Any]]:
        """
        Procesa múltiples facturas y guarda resultados en JSON
        """
        results = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp'}

        images_path = Path(images_directory)
        image_files = [f for f in images_path.iterdir()
                      if f.suffix.lower() in image_extensions]

        print(f"📋 Procesando {len(image_files)} facturas...")

        for image_file in image_files:
            try:
                result = self.process_invoice(str(image_file))
                results.append(result)
                print(f"✅ Procesado: {image_file.name}")
            except Exception as e:
                print(f"❌ Error procesando {image_file.name}: {e}")
                results.append({
                    "error": str(e),
                    "archivo_procesado": image_file.name
                })

        # Guardar resultados
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Resultados guardados en: {output_file}")
        return results

def main():
    """Función principal de ejemplo"""
    processor = QwenInvoiceProcessor()

    # Procesar una factura individual
    test_image = "./test_invoices/TELECOM 4264-3649020.pdf"
    if os.path.exists(test_image):
        result = processor.process_invoice(test_image)
        print("\n" + "="*60)
        print("📋 RESULTADO EXTRAÍDO:")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("="*60)

    # Procesar batch si existe el directorio
    if os.path.exists("./test_invoices"):
        print("\n🔄 Procesando batch de facturas...")
        results = processor.process_batch("./test_invoices")
        print(f"✅ Procesadas {len(results)} facturas")

if __name__ == "__main__":
    main()