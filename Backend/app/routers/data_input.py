# backend/app/routers/data_input.py

from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from typing import List, Optional
import logging
import base64

from app.services import data_service
from app.models.requests import ConfigureMetadataRequest, ReplaceImageRequest

router = APIRouter()

@router.post("/images")
async def upload_images(
    images: Optional[List[UploadFile]] = File(None, description="Imagens soltas"),
    zip_file: Optional[UploadFile] = File(None, description="Zip contendo imagens"),
    session_id: Optional[str] = Query(None, description="Optional session id for per-user in-memory storage")
):
    try:
        saved_files = []

        # 1) imagens avulsas
        if images:
            # Save into session if session_id provided, otherwise persist
            saved = await data_service.save_uploaded_images(images, session_id=session_id)
            saved_files.extend(saved)
        # 2) ZIP
        if zip_file:
            if not zip_file.filename.lower().endswith(".zip"):
                raise HTTPException(status_code=400, detail="zip_file deve ser .zip")
            extracted_paths = await data_service.save_and_extract_zip(zip_file, session_id=session_id)
            saved_files.extend(extracted_paths)

        if not saved_files:
            raise HTTPException(status_code=400, detail="Nenhum arquivo válido enviado")

        return {"image_ids": saved_files}
    except Exception as e:
        logging.exception("Failed to upload images")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/metadata")
async def upload_metadata(
    file: UploadFile,
    delimiter: str = Form(","),
    decimal_sep: str = Form(".")
):
    try:
        columns = await data_service.read_metadata(file, delimiter, decimal_sep)
        return {"columns": columns}
    except Exception as e:
        logging.exception("Failed to upload metadata")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/metadata/configure")
async def configure_metadata(req: ConfigureMetadataRequest):
    try:
        await data_service.configure_metadata(req.image_id_col, req.col_mapping)
        return {"status": "success"}
    except Exception as e:
        logging.exception("Failed to configure metadata")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/replace-image")
async def replace_image(request: ReplaceImageRequest, session_id: Optional[str] = Query(None)):
    """
    Replace an existing image on the backend with a new (cropped) version.
    The replacement is persistent and affects all downstream processing.
    """
    try:
        logging.info(f"Received request to replace image: {request.image_id}")
        if session_id:
            data_service.replace_image_in_session(session_id, request.image_id, request.image_data_base64)
        else:
            data_service.replace_image(request.image_id, request.image_data_base64)
        logging.info("Image replacement successful")
        return {"status": "success"}
    except FileNotFoundError as e:
        logging.error(f"Image not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logging.error(f"Invalid image data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        logging.error(f"Failed to save image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error replacing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# -----------------------------
# New: Extract images from CSV of URLs
# -----------------------------
@router.post("/extract-from-csv")
async def extract_from_csv(file: UploadFile = File(...), session_id: Optional[str] = Query(None)):
    """
    Accept a CSV file where each row contains an image URL.
    Downloads all images, stores them in IMAGE_DIR (clearing old ones),
    and returns a base64-encoded ZIP plus the list of saved image IDs and any errors.
    """
    try:
        result = await data_service.extract_images_from_csv(file)
        zip_b64 = base64.b64encode(result["zip_bytes"]).decode("utf-8")
        return {
            "zip_b64": zip_b64,
            "image_ids": result.get("image_ids", []),
            "errors": result.get("errors", []),
        }
    except ValueError as e:
        logging.error(f"Bad CSV for extractor: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Extractor failed")
        raise HTTPException(status_code=500, detail="Internal server error during extraction")


@router.get("/current-image-ids")
async def get_current_image_ids(session_id: Optional[str] = Query(None)):
    """Get the current image IDs in the order they are stored in the backend.

    If session_id is provided, return only that session's images.
    """
    try:
        image_ids = data_service.get_all_image_ids(session_id=session_id)
        return {"image_ids": image_ids}
    except Exception as e:
        logging.exception("Failed to get image IDs")
        raise HTTPException(status_code=500, detail="Failed to retrieve image IDs")


@router.get("/image/{image_id}")
async def get_image(image_id: str, session_id: Optional[str] = Query(None)):
    """Get a single image by its ID. If session_id is provided, look up in-memory session store."""
    try:
        image_bytes = data_service.load_image(image_id, session_id=session_id)
        return Response(content=image_bytes, media_type="image/png")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
    except Exception as e:
        logging.exception("Failed to retrieve image")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")


@router.delete("/clear-all-images")
async def clear_all_images(session_id: Optional[str] = Query(None)):
    """Clear stored images. If session_id provided, clear only that session's images."""
    try:
        data_service.clear_all_images(session_id=session_id)
        return {"message": "Images cleared successfully"}
    except Exception as e:
        logging.exception("Failed to clear images")
        raise HTTPException(status_code=500, detail="Failed to clear images")
