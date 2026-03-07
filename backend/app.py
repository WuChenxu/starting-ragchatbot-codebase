import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import os
from typing import List, Optional

from config import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from models import Source
from pydantic import BaseModel
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""

    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    """Source with display text and optional link"""

    text: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: List[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: List[str]


# API Endpoints


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)

        # Convert Source objects to SourceItem for response
        source_items = [SourceItem(text=source.text, link=source.link) for source in sources]

        return QueryResponse(answer=answer, sources=source_items, session_id=session_id)
    except Exception as e:
        error_msg = str(e)
        print(f"Query error: {error_msg}")

        # Provide more user-friendly error messages
        if "Invalid Moonshot API Key" in error_msg or "Authentication" in error_msg:
            raise HTTPException(
                status_code=401,
                detail="API Key 无效。请在 .env 文件中设置正确的 MOONSHOT_API_KEY，然后重启服务器。获取 API Key: https://platform.moonshot.cn/",
            )

        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"], course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteSessionRequest(BaseModel):
    """Request model for deleting a session"""

    session_id: str


class DeleteSessionResponse(BaseModel):
    """Response model for session deletion"""

    success: bool
    message: str


@app.post("/api/session/delete", response_model=DeleteSessionResponse)
async def delete_session(request: DeleteSessionRequest):
    """Delete a conversation session"""
    try:
        success = rag_system.session_manager.delete_session(request.session_id)
        if success:
            return DeleteSessionResponse(
                success=True, message=f"Session {request.session_id} deleted successfully"
            )
        else:
            return DeleteSessionResponse(
                success=False, message=f"Session {request.session_id} not found"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            print(f"Error loading documents: {e}")


import os
from pathlib import Path

from fastapi.responses import FileResponse

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
