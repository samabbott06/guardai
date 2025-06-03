"""
Response models for structured output from Ollama.
These models are used by the application classes to parse the output from Ollama
into a structured format specific to each application type.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel

class EmailResponse(BaseModel):
    """
    Model for email client assistant response.
    """
    email_summary: str
    actions: Optional[List[Dict]] = None

    def to_dict(self):
        """
        Convert the response to a dictionary.
        
        Returns:
            Dictionary representation of the response
        """
        return {
            "summary": self.email_summary,
            "actions": self.actions or []
        }

class TechSupportResponse(BaseModel):
    """
    Model for tech support chatbot response.
    """
    explanation: str
    commands: Optional[List[str]] = None
    
    def to_dict(self):
        """
        Convert the response to a dictionary.
        
        Returns:
            Dictionary representation of the response
        """
        return {
            "explanation": self.explanation,
            "commands": self.commands or []
        }

class DocumentSummaryResponse(BaseModel):
    """
    Model for document summarization service response.
    """
    summary: str
    key_points: Optional[List[str]] = None
    
    def to_dict(self):
        """
        Convert the response to a dictionary.
        
        Returns:
            Dictionary representation of the response
        """
        return {
            "summary": self.summary,
            "key_points": self.key_points or []
        }

class PersonalAssistantResponse(BaseModel):
    """
    Model for personal assistant response.
    """
    response: str
    actions: Optional[List[Dict]] = None
    
    def to_dict(self):
        """
        Convert the response to a dictionary.
        
        Returns:
            Dictionary representation of the response
        """
        return {
            "response": self.response,
            "actions": self.actions or []
        }
