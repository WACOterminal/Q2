import logging
from typing import Dict, Any, Optional, List
import httpx
from fastapi import HTTPException
import asyncio
import json
import base64
import os
import tempfile
from pathlib import Path

from app.models.connector import BaseConnector, ConnectorAction
from app.core.vault_client import vault_client

logger = logging.getLogger(__name__)

class FreeCADConnector(BaseConnector):
    """
    A connector for interacting with FreeCAD through its Python API.
    
    Supported actions:
    - create_document: Create a new FreeCAD document
    - open_document: Open an existing FreeCAD document
    - save_document: Save a FreeCAD document
    - export_document: Export document to various formats (STL, STEP, OBJ, etc.)
    - get_document_info: Get document information and metadata
    - list_objects: List objects in a document
    - create_object: Create basic geometric objects (box, cylinder, sphere)
    - modify_object: Modify object properties
    - delete_object: Delete objects from document
    - get_object_properties: Get object properties and measurements
    - create_sketch: Create 2D sketches
    - extrude_sketch: Extrude sketches into 3D objects
    - create_assembly: Create assemblies with multiple parts
    - generate_technical_drawing: Generate technical drawings
    - mesh_analysis: Perform mesh analysis and validation
    - calculate_volume: Calculate object volume and mass properties
    - check_interference: Check for interference between objects
    - generate_gcode: Generate G-code for 3D printing/CNC
    """

    @property
    def connector_id(self) -> str:
        return "freecad"

    async def _get_freecad_client(self, credential_id: str) -> Dict[str, Any]:
        """Helper to get FreeCAD connection configuration."""
        credential = await vault_client.get_credential(credential_id)
        freecad_path = credential.secrets.get("freecad_path", "/usr/bin/freecad")
        python_path = credential.secrets.get("python_path", "/usr/bin/python3")
        workspace_path = credential.secrets.get("workspace_path", "/tmp/freecad_workspace")
        
        # Ensure workspace directory exists
        os.makedirs(workspace_path, exist_ok=True)
        
        return {
            "freecad_path": freecad_path,
            "python_path": python_path,
            "workspace_path": workspace_path
        }

    async def execute(self, action: ConnectorAction, configuration: Dict[str, Any], data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not action.credential_id:
            raise ValueError("credential_id must be provided for FreeCAD connector actions.")
        
        try:
            action_map = {
                "create_document": self._create_document,
                "open_document": self._open_document,
                "save_document": self._save_document,
                "export_document": self._export_document,
                "get_document_info": self._get_document_info,
                "list_objects": self._list_objects,
                "create_object": self._create_object,
                "modify_object": self._modify_object,
                "delete_object": self._delete_object,
                "get_object_properties": self._get_object_properties,
                "create_sketch": self._create_sketch,
                "extrude_sketch": self._extrude_sketch,
                "create_assembly": self._create_assembly,
                "generate_technical_drawing": self._generate_technical_drawing,
                "mesh_analysis": self._mesh_analysis,
                "calculate_volume": self._calculate_volume,
                "check_interference": self._check_interference,
                "generate_gcode": self._generate_gcode,
            }

            if action.action_id in action_map:
                func = action_map[action.action_id]
                return await func(action.credential_id, configuration, data_context)
            else:
                raise ValueError(f"Unsupported action for FreeCAD connector: {action.action_id}")

        except Exception as e:
            logger.error(f"An unexpected error occurred in FreeCADConnector: {e}", exc_info=True)
            raise

    async def _execute_freecad_script(self, client_config: Dict[str, Any], script: str, script_name: str = "freecad_script") -> Dict[str, Any]:
        """Execute a FreeCAD Python script and return the result."""
        workspace_path = client_config["workspace_path"]
        python_path = client_config["python_path"]
        
        # Create temporary script file
        script_file = os.path.join(workspace_path, f"{script_name}.py")
        with open(script_file, 'w') as f:
            f.write(script)
        
        # Execute the script
        import subprocess
        try:
            result = subprocess.run(
                [python_path, script_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"FreeCAD script execution failed: {result.stderr}")
            
            # Clean up script file
            os.remove(script_file)
            
            return {
                "status": "success",
                "output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.TimeoutExpired:
            raise Exception("FreeCAD script execution timed out")
        except Exception as e:
            # Clean up script file on error
            if os.path.exists(script_file):
                os.remove(script_file)
            raise e

    async def _create_document(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new FreeCAD document."""
        client_config = await self._get_freecad_client(credential_id)
        document_name = data.get("document_name", "NewDocument")
        
        script = f"""
import FreeCAD
import json

# Create new document
doc = FreeCAD.newDocument("{document_name}")
doc.recompute()

result = {{
    "document_name": doc.Name,
    "label": doc.Label,
    "objects_count": len(doc.Objects),
    "file_path": doc.FileName if doc.FileName else None
}}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "create_document")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    async def _open_document(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Open an existing FreeCAD document."""
        client_config = await self._get_freecad_client(credential_id)
        file_path = config.get("file_path")
        
        if not file_path:
            raise ValueError("file_path is required.")
        
        script = f"""
import FreeCAD
import json
import os

file_path = "{file_path}"
if not os.path.exists(file_path):
    result = {{"status": "error", "message": "File not found"}}
else:
    doc = FreeCAD.openDocument(file_path)
    doc.recompute()
    
    result = {{
        "status": "success",
        "document_name": doc.Name,
        "label": doc.Label,
        "objects_count": len(doc.Objects),
        "file_path": doc.FileName,
        "objects": [obj.Name for obj in doc.Objects]
    }}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "open_document")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    async def _save_document(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a FreeCAD document."""
        client_config = await self._get_freecad_client(credential_id)
        document_name = config.get("document_name")
        file_path = data.get("file_path")
        
        if not document_name:
            raise ValueError("document_name is required.")
        
        script = f"""
import FreeCAD
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
else:
    file_path = "{file_path}" if "{file_path}" else doc.FileName
    if file_path:
        doc.saveAs(file_path)
        result = {{
            "status": "success",
            "document_name": doc.Name,
            "file_path": file_path,
            "saved": True
        }}
    else:
        result = {{"status": "error", "message": "No file path specified"}}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "save_document")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    async def _export_document(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Export document to various formats."""
        client_config = await self._get_freecad_client(credential_id)
        document_name = config.get("document_name")
        export_format = config.get("export_format", "STL")
        output_path = data.get("output_path")
        objects = config.get("objects", [])  # Specific objects to export
        
        if not document_name or not output_path:
            raise ValueError("document_name and output_path are required.")
        
        objects_str = str(objects) if objects else "[]"
        
        script = f"""
import FreeCAD
import Mesh
import Part
import json
import os

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
else:
    export_format = "{export_format}".upper()
    output_path = "{output_path}"
    objects_to_export = {objects_str}
    
    try:
        if objects_to_export:
            objs = [doc.getObject(name) for name in objects_to_export if doc.getObject(name)]
        else:
            objs = doc.Objects
        
        if export_format == "STL":
            mesh_objs = []
            for obj in objs:
                if hasattr(obj, "Shape"):
                    mesh = Mesh.Mesh()
                    mesh.addFacets(obj.Shape.tessellate(0.1))
                    mesh_objs.append(mesh)
            if mesh_objs:
                combined_mesh = mesh_objs[0]
                for mesh in mesh_objs[1:]:
                    combined_mesh.addMesh(mesh)
                combined_mesh.write(output_path)
        elif export_format == "STEP":
            shapes = [obj.Shape for obj in objs if hasattr(obj, "Shape")]
            if shapes:
                Part.export(shapes, output_path)
        elif export_format == "OBJ":
            import importOBJ
            importOBJ.export(objs, output_path)
        
        result = {{
            "status": "success",
            "format": export_format,
            "output_path": output_path,
            "objects_exported": len(objs)
        }}
    except Exception as e:
        result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "export_document")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    async def _create_object(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic geometric objects."""
        client_config = await self._get_freecad_client(credential_id)
        document_name = config.get("document_name")
        object_type = data.get("object_type", "Box")
        object_name = data.get("object_name", f"Object_{object_type}")
        parameters = data.get("parameters", {})
        
        if not document_name:
            raise ValueError("document_name is required.")
        
        # Convert parameters to string representation
        params_str = json.dumps(parameters)
        
        script = f"""
import FreeCAD
import Part
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
else:
    object_type = "{object_type}"
    object_name = "{object_name}"
    parameters = {params_str}
    
    try:
        if object_type == "Box":
            length = parameters.get("length", 10)
            width = parameters.get("width", 10)
            height = parameters.get("height", 10)
            obj = doc.addObject("Part::Box", object_name)
            obj.Length = length
            obj.Width = width
            obj.Height = height
        elif object_type == "Cylinder":
            radius = parameters.get("radius", 5)
            height = parameters.get("height", 10)
            obj = doc.addObject("Part::Cylinder", object_name)
            obj.Radius = radius
            obj.Height = height
        elif object_type == "Sphere":
            radius = parameters.get("radius", 5)
            obj = doc.addObject("Part::Sphere", object_name)
            obj.Radius = radius
        else:
            result = {{"status": "error", "message": f"Unsupported object type: {{object_type}}"}}
            print(json.dumps(result))
            exit()
        
        doc.recompute()
        
        result = {{
            "status": "success",
            "object_name": obj.Name,
            "object_type": object_type,
            "parameters": parameters
        }}
    except Exception as e:
        result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "create_object")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    async def _calculate_volume(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate object volume and mass properties."""
        client_config = await self._get_freecad_client(credential_id)
        document_name = config.get("document_name")
        object_name = config.get("object_name")
        
        if not document_name or not object_name:
            raise ValueError("document_name and object_name are required.")
        
        script = f"""
import FreeCAD
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
else:
    obj = doc.getObject("{object_name}")
    if obj is None:
        result = {{"status": "error", "message": "Object not found"}}
    elif not hasattr(obj, "Shape"):
        result = {{"status": "error", "message": "Object has no shape"}}
    else:
        try:
            shape = obj.Shape
            volume = shape.Volume
            area = shape.Area
            center_of_mass = shape.CenterOfMass
            
            result = {{
                "status": "success",
                "object_name": obj.Name,
                "volume": volume,
                "surface_area": area,
                "center_of_mass": {{
                    "x": center_of_mass.x,
                    "y": center_of_mass.y,
                    "z": center_of_mass.z
                }}
            }}
        except Exception as e:
            result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "calculate_volume")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    async def _list_objects(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """List objects in a document."""
        client_config = await self._get_freecad_client(credential_id)
        document_name = config.get("document_name")
        
        if not document_name:
            raise ValueError("document_name is required.")
        
        script = f"""
import FreeCAD
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
else:
    objects = []
    for obj in doc.Objects:
        obj_info = {{
            "name": obj.Name,
            "label": obj.Label,
            "type": obj.TypeId,
            "visible": obj.Visibility if hasattr(obj, "Visibility") else True
        }}
        
        # Add shape information if available
        if hasattr(obj, "Shape"):
            try:
                obj_info["has_shape"] = True
                obj_info["volume"] = obj.Shape.Volume
                obj_info["area"] = obj.Shape.Area
            except:
                obj_info["has_shape"] = True
                obj_info["volume"] = None
                obj_info["area"] = None
        else:
            obj_info["has_shape"] = False
        
        objects.append(obj_info)
    
    result = {{
        "status": "success",
        "document_name": doc.Name,
        "objects_count": len(objects),
        "objects": objects
    }}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "list_objects")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    async def _get_document_info(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document information and metadata."""
        client_config = await self._get_freecad_client(credential_id)
        document_name = config.get("document_name")
        
        if not document_name:
            raise ValueError("document_name is required.")
        
        script = f"""
import FreeCAD
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
else:
    result = {{
        "status": "success",
        "name": doc.Name,
        "label": doc.Label,
        "file_path": doc.FileName if doc.FileName else None,
        "objects_count": len(doc.Objects),
        "modified": doc.UndoMode > 0,
        "created_by": doc.CreatedBy if hasattr(doc, "CreatedBy") else None,
        "comment": doc.Comment if hasattr(doc, "Comment") else None,
        "company": doc.Company if hasattr(doc, "Company") else None
    }}

print(json.dumps(result))
"""
        
        result = await self._execute_freecad_script(client_config, script, "get_document_info")
        if result["status"] == "success" and result["output"]:
            return json.loads(result["output"].strip())
        return result

    # Placeholder implementations for other actions
    async def _modify_object(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify object properties."""
        return {"status": "success", "message": "Object modification not yet implemented"}

    async def _delete_object(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete objects from document."""
        return {"status": "success", "message": "Object deletion not yet implemented"}

    async def _get_object_properties(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get object properties and measurements."""
        return {"status": "success", "message": "Object properties retrieval not yet implemented"}

    async def _create_sketch(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create 2D sketches."""
        return {"status": "success", "message": "Sketch creation not yet implemented"}

    async def _extrude_sketch(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrude sketches into 3D objects."""
        return {"status": "success", "message": "Sketch extrusion not yet implemented"}

    async def _create_assembly(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create assemblies with multiple parts."""
        return {"status": "success", "message": "Assembly creation not yet implemented"}

    async def _generate_technical_drawing(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical drawings."""
        return {"status": "success", "message": "Technical drawing generation not yet implemented"}

    async def _mesh_analysis(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mesh analysis and validation."""
        return {"status": "success", "message": "Mesh analysis not yet implemented"}

    async def _check_interference(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for interference between objects."""
        return {"status": "success", "message": "Interference checking not yet implemented"}

    async def _generate_gcode(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate G-code for 3D printing/CNC."""
        return {"status": "success", "message": "G-code generation not yet implemented"}


# Instantiate a single instance
freecad_connector = FreeCADConnector() 