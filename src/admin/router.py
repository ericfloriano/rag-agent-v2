import os
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Request, Response, Depends, Form, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.retrieval.ingest import ingest_all
from src.retrieval.vector_store import get_qdrant_client
from src.config import COLLECTION_NAME, CACHE_COLLECTION_NAME, ADMIN_SECRET_KEY, TURNSTILE_SITEKEY
from src.admin.security import (
    limiter,
    verify_admin_session,
    verify_turnstile,
    EXPECTED_COOKIE_HASH,
    COOKIE_NAME,
)

# Template Engine Setup
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

router = APIRouter(prefix="/admin", tags=["Admin Panel"])


@router.get("/login", response_class=HTMLResponse)
@limiter.limit("10/minute") # Rate Limiting (OWASP A07)
async def login_page(request: Request):
    """Render the secure login page with Anti-Bot protection."""
    return templates.TemplateResponse("login.html", {"request": request, "sitekey": TURNSTILE_SITEKEY})


@router.post("/login")
@limiter.limit("5/minute") # Strict Rate Limiting (Anti Brute-Force)
async def login_submit(
    request: Request,
    password: str = Form(...),
    cf_turnstile_response: str = Form(""),
):
    """Process login, call Cloudflare, return HttpOnly cookie."""
    
    # 1. Cloudflare Anti-Bot Challenge
    if not await verify_turnstile(cf_turnstile_response, request.client.host):
        return templates.TemplateResponse(
            "login.html", 
            {"request": request, "error": "Anti-bot verification failed.", "sitekey": TURNSTILE_SITEKEY},
            status_code=400
        )
    
    # 2. Hardened Password Check
    if password != ADMIN_SECRET_KEY:
        return templates.TemplateResponse(
            "login.html", 
            {"request": request, "error": "Invalid credentials.", "sitekey": TURNSTILE_SITEKEY},
            status_code=401
        )
        
    # 3. Successful Login: Set secure OWASP-compliant cookie (A01)
    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        key=COOKIE_NAME,
        value=EXPECTED_COOKIE_HASH,
        httponly=True,  # Blocks XSS theft
        secure=request.url.scheme == "https",  # Enforce HTTPS transmission
        samesite="lax", # Anti-CSRF
        max_age=3600  # 1 Hour lifespan
    )
    return response


@router.get("/logout")
async def logout():
    """Clear session safely."""
    response = RedirectResponse(url="/admin/login")
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Main Admin Dashboard (Protected Route).
    Lists uploaded PDFs and statistics directly from Qdrant.
    """
    try:
        await verify_admin_session(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)

    client = get_qdrant_client()
    
    docs_summary = {}
    try:
        from qdrant_client.http.models import Filter
        # Scroll logic: get a sample of 1000 points strictly to aggregate metadata
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        # Aggregate chunk stats by PDF filename
        for p in points:
            fname = p.payload.get("metadata", {}).get("source_filename", "unknown.pdf")
            docs_summary[fname] = docs_summary.get(fname, 0) + 1
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not load vector details: {e}")

    return templates.TemplateResponse("dashboard.html", {"request": request, "documents": docs_summary})


@router.post("/upload", dependencies=[Depends(verify_admin_session)])
@limiter.limit("10/minute") # Rate Limiter on expensive operation
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Accepts PDF upload, runs Langchain split & Qdrant ingestion (Zero-Cost).
    Input Validation (OWASP A03).
    """
    # 1. Input Strict Validation
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT documents are allowed.")
        
    # Validating File Size (Max 15MB to prevent DoS Out-of-Memory)
    if file.size and file.size > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The file exceeds the 15MB limit.")

    # 2. Local ingestion pipeline
    import shutil
    try:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / file.filename
            
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Repurpose existing bulk script to ingest this single directory
            chunks_ingested = ingest_all(documents_dir=Path(temp_dir))
            
            return {"status": "success", "message": f"File {file.filename} ingested! {chunks_ingested} vectors stored."}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error ingesting Admin Document: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the document.")


@router.delete("/documents/{filename}", dependencies=[Depends(verify_admin_session)])
async def delete_document(request: Request, filename: str):
    """Delete all chunks related to a specific PDF from Qdrant Cloud."""
    client = get_qdrant_client()
    
    # Ensure index exists before trying to filter
    from src.retrieval.vector_store import create_payload_indexes_if_needed
    create_payload_indexes_if_needed(client)
    
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    try:
        # Match point by metadata
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_filename", 
                        match=MatchValue(value=filename)
                    )
                ]
            )
        )
        return {"status": "success", "message": f"Document '{filename}' permanently purged."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache", dependencies=[Depends(verify_admin_session)])
async def clear_semantic_cache(request: Request):
    """
    Zero-Cost Semantic Cache Wipeout.
    Deletes points internally in the cache collection.
    """
    client = get_qdrant_client()
    try:
        from qdrant_client.http.models import Filter
        client.delete(
            collection_name=CACHE_COLLECTION_NAME,
            points_selector=Filter() # Empty generic filter wipes entire collection contents
        )
        return {"status": "success", "message": "Semantic Cache successfully purged!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
