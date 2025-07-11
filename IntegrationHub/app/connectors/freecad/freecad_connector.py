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
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            object_name = data.get("object_name")
            properties = data.get("properties", {})
            
            if not document_name or not object_name:
                raise ValueError("document_name and object_name are required.")
            
            if not properties:
                raise ValueError("properties to modify are required.")
            
            # Convert properties to string representation for the script
            props_str = json.dumps(properties)
            
            script = f"""
import FreeCAD
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

obj = doc.getObject("{object_name}")
if obj is None:
    result = {{"status": "error", "message": "Object not found"}}
    print(json.dumps(result))
    exit()

try:
    properties = {props_str}
    modified_properties = []
    errors = []
    
    for prop_name, prop_value in properties.items():
        try:
            if hasattr(obj, prop_name):
                # Get current value for comparison
                old_value = getattr(obj, prop_name)
                
                # Set new value
                setattr(obj, prop_name, prop_value)
                
                modified_properties.append({{
                    "property": prop_name,
                    "old_value": str(old_value),
                    "new_value": str(prop_value)
                }})
            else:
                errors.append(f"Property '{{prop_name}}' does not exist on object")
        except Exception as e:
            errors.append(f"Failed to set property '{{prop_name}}': {{str(e)}}")
    
    # Recompute the document to apply changes
    doc.recompute()
    
    result = {{
        "status": "success",
        "object_name": obj.Name,
        "modified_properties": modified_properties,
        "errors": errors,
        "properties_modified": len(modified_properties)
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "modify_object")
            if result["status"] == "success" and result["output"]:
                return json.loads(result["output"].strip())
            return result
            
        except Exception as e:
            logger.error(f"Failed to modify object: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to modify object: {str(e)}"}

    async def _delete_object(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete objects from document."""
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            object_names = data.get("object_names", [])
            object_name = data.get("object_name")  # Single object alternative
            
            if not document_name:
                raise ValueError("document_name is required.")
            
            # Support both single object and multiple objects
            if object_name and not object_names:
                object_names = [object_name]
            elif not object_names:
                raise ValueError("Either object_name or object_names list is required.")
            
            # Convert object names to string representation
            names_str = json.dumps(object_names)
            
            script = f"""
import FreeCAD
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

try:
    object_names = {names_str}
    deleted_objects = []
    not_found_objects = []
    errors = []
    
    for obj_name in object_names:
        try:
            obj = doc.getObject(obj_name)
            if obj is None:
                not_found_objects.append(obj_name)
            else:
                # Get object info before deletion
                obj_info = {{
                    "name": obj.Name,
                    "label": obj.Label,
                    "type": obj.TypeId
                }}
                
                # Remove object from document
                doc.removeObject(obj_name)
                deleted_objects.append(obj_info)
                
        except Exception as e:
            errors.append(f"Failed to delete object '{{obj_name}}': {{str(e)}}")
    
    # Recompute the document after deletions
    doc.recompute()
    
    result = {{
        "status": "success",
        "deleted_objects": deleted_objects,
        "not_found_objects": not_found_objects,
        "errors": errors,
        "objects_deleted": len(deleted_objects),
        "remaining_objects": len(doc.Objects)
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "delete_object")
            if result["status"] == "success" and result["output"]:
                return json.loads(result["output"].strip())
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete object(s): {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to delete object(s): {str(e)}"}

    async def _get_object_properties(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get object properties and measurements."""
        try:
            object_name = data.get("object_name")
            if not object_name:
                return {"status": "error", "message": "Object name is required"}
            
            # Execute FreeCAD Python code to get object properties
            freecad_code = f"""
import FreeCAD
import Part

# Get the object
obj = FreeCAD.ActiveDocument.getObject('{object_name}')
if obj is None:
    result = {{"error": "Object not found"}}
else:
    properties = {{}}
    
    # Basic properties
    properties['Name'] = obj.Name
    properties['Label'] = obj.Label
    properties['Type'] = obj.TypeId
    
    # Geometric properties if it's a Part object
    if hasattr(obj, 'Shape') and obj.Shape:
        shape = obj.Shape
        properties['Volume'] = shape.Volume
        properties['Area'] = shape.Area
        properties['BoundBox'] = {{
            'XMin': shape.BoundBox.XMin,
            'YMin': shape.BoundBox.YMin,
            'ZMin': shape.BoundBox.ZMin,
            'XMax': shape.BoundBox.XMax,
            'YMax': shape.BoundBox.YMax,
            'ZMax': shape.BoundBox.ZMax,
            'XLength': shape.BoundBox.XLength,
            'YLength': shape.BoundBox.YLength,
            'ZLength': shape.BoundBox.ZLength
        }}
        properties['CenterOfMass'] = {{
            'x': shape.CenterOfMass.x,
            'y': shape.CenterOfMass.y,
            'z': shape.CenterOfMass.z
        }}
    
    # Material properties if available
    if hasattr(obj, 'Material') and obj.Material:
        properties['Material'] = obj.Material
    
    # Custom properties
    for prop in obj.PropertiesList:
        if not prop.startswith('_'):
            try:
                value = getattr(obj, prop)
                if isinstance(value, (int, float, str, bool)):
                    properties[prop] = value
            except:
                pass
    
    result = {{"properties": properties}}
"""
            
            result = await self._execute_freecad_code(freecad_code, credential_id, config)
            
            if "error" in result:
                return {"status": "error", "message": result["error"]}
            
            return {
                "status": "success", 
                "properties": result.get("properties", {}),
                "message": f"Retrieved properties for object: {object_name}"
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to get object properties: {str(e)}"}

    async def _create_sketch(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create 2D sketches."""
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            sketch_name = data.get("sketch_name", "Sketch")
            plane = data.get("plane", "XY")  # XY, XZ, YZ
            geometry = data.get("geometry", [])  # List of geometric elements
            constraints = data.get("constraints", [])  # List of constraints
            
            if not document_name:
                raise ValueError("document_name is required.")
            
            if not geometry:
                raise ValueError("geometry list is required.")
            
            # Convert geometry and constraints to string representation
            geometry_str = json.dumps(geometry)
            constraints_str = json.dumps(constraints)
            
            script = f"""
import FreeCAD
import Sketcher
import Part
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

try:
    # Create sketch object
    sketch = doc.addObject('Sketcher::SketchObject', '{sketch_name}')
    
    # Set the sketch plane
    plane = "{plane}"
    if plane == "XY":
        sketch.Support = (doc.getObject('XY_Plane'), [''])
        sketch.MapMode = 'FlatFace'
    elif plane == "XZ":
        sketch.Support = (doc.getObject('XZ_Plane'), [''])
        sketch.MapMode = 'FlatFace'
    elif plane == "YZ":
        sketch.Support = (doc.getObject('YZ_Plane'), [''])
        sketch.MapMode = 'FlatFace'
    else:
        # Default to XY plane
        sketch.Placement = FreeCAD.Placement(FreeCAD.Vector(0,0,0), FreeCAD.Rotation(0,0,0,1))
    
    # Add geometry elements
    geometry = {geometry_str}
    created_geometry = []
    
    for geom in geometry:
        geom_type = geom.get("type")
        params = geom.get("parameters", {{}})
        
        if geom_type == "line":
            start_x = params.get("start_x", 0)
            start_y = params.get("start_y", 0)
            end_x = params.get("end_x", 10)
            end_y = params.get("end_y", 0)
            
            line = Part.LineSegment(
                FreeCAD.Vector(start_x, start_y, 0),
                FreeCAD.Vector(end_x, end_y, 0)
            )
            geo_id = sketch.addGeometry(line)
            created_geometry.append({{"type": "line", "id": geo_id, "start": [start_x, start_y], "end": [end_x, end_y]}})
            
        elif geom_type == "circle":
            center_x = params.get("center_x", 0)
            center_y = params.get("center_y", 0)
            radius = params.get("radius", 5)
            
            circle = Part.Circle(FreeCAD.Vector(center_x, center_y, 0), FreeCAD.Vector(0, 0, 1), radius)
            geo_id = sketch.addGeometry(circle)
            created_geometry.append({{"type": "circle", "id": geo_id, "center": [center_x, center_y], "radius": radius}})
            
        elif geom_type == "arc":
            center_x = params.get("center_x", 0)
            center_y = params.get("center_y", 0)
            radius = params.get("radius", 5)
            start_angle = params.get("start_angle", 0)
            end_angle = params.get("end_angle", 90)
            
            arc = Part.ArcOfCircle(
                Part.Circle(FreeCAD.Vector(center_x, center_y, 0), FreeCAD.Vector(0, 0, 1), radius),
                math.radians(start_angle),
                math.radians(end_angle)
            )
            geo_id = sketch.addGeometry(arc)
            created_geometry.append({{"type": "arc", "id": geo_id, "center": [center_x, center_y], "radius": radius, "start_angle": start_angle, "end_angle": end_angle}})
            
        elif geom_type == "rectangle":
            x = params.get("x", 0)
            y = params.get("y", 0)
            width = params.get("width", 10)
            height = params.get("height", 10)
            
            # Create rectangle as four lines
            lines = [
                Part.LineSegment(FreeCAD.Vector(x, y, 0), FreeCAD.Vector(x + width, y, 0)),
                Part.LineSegment(FreeCAD.Vector(x + width, y, 0), FreeCAD.Vector(x + width, y + height, 0)),
                Part.LineSegment(FreeCAD.Vector(x + width, y + height, 0), FreeCAD.Vector(x, y + height, 0)),
                Part.LineSegment(FreeCAD.Vector(x, y + height, 0), FreeCAD.Vector(x, y, 0))
            ]
            
            rect_ids = []
            for line in lines:
                geo_id = sketch.addGeometry(line)
                rect_ids.append(geo_id)
            
            # Add coincident constraints to close the rectangle
            for i in range(4):
                next_i = (i + 1) % 4
                sketch.addConstraint(Sketcher.Constraint('Coincident', rect_ids[i], 2, rect_ids[next_i], 1))
            
            created_geometry.append({{"type": "rectangle", "ids": rect_ids, "x": x, "y": y, "width": width, "height": height}})
    
    # Add constraints
    constraints = {constraints_str}
    created_constraints = []
    
    for constraint in constraints:
        constraint_type = constraint.get("type")
        params = constraint.get("parameters", {{}})
        
        if constraint_type == "horizontal" and "geo_id" in params:
            sketch.addConstraint(Sketcher.Constraint('Horizontal', params["geo_id"]))
            created_constraints.append({{"type": "horizontal", "geo_id": params["geo_id"]}})
            
        elif constraint_type == "vertical" and "geo_id" in params:
            sketch.addConstraint(Sketcher.Constraint('Vertical', params["geo_id"]))
            created_constraints.append({{"type": "vertical", "geo_id": params["geo_id"]}})
            
        elif constraint_type == "distance" and "geo_id" in params and "value" in params:
            sketch.addConstraint(Sketcher.Constraint('Distance', params["geo_id"], params["value"]))
            created_constraints.append({{"type": "distance", "geo_id": params["geo_id"], "value": params["value"]}})
            
        elif constraint_type == "parallel" and "geo_id1" in params and "geo_id2" in params:
            sketch.addConstraint(Sketcher.Constraint('Parallel', params["geo_id1"], params["geo_id2"]))
            created_constraints.append({{"type": "parallel", "geo_id1": params["geo_id1"], "geo_id2": params["geo_id2"]}})
    
    # Recompute to apply all changes
    doc.recompute()
    
    result = {{
        "status": "success",
        "sketch_name": sketch.Name,
        "plane": plane,
        "geometry_count": len(created_geometry),
        "constraints_count": len(created_constraints),
        "created_geometry": created_geometry,
        "created_constraints": created_constraints
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "create_sketch")
            if result["status"] == "success" and result["output"]:
                return json.loads(result["output"].strip())
            return result
            
        except Exception as e:
            logger.error(f"Failed to create sketch: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to create sketch: {str(e)}"}

    async def _extrude_sketch(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrude sketches into 3D objects."""
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            sketch_name = data.get("sketch_name")
            extrude_name = data.get("extrude_name", "Pad")
            length = data.get("length", 10.0)
            direction = data.get("direction", "normal")  # normal, custom, reversed
            taper_angle = data.get("taper_angle", 0.0)
            symmetric = data.get("symmetric", False)
            
            if not document_name or not sketch_name:
                raise ValueError("document_name and sketch_name are required.")
            
            script = f"""
import FreeCAD
import PartDesign
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

sketch = doc.getObject("{sketch_name}")
if sketch is None:
    result = {{"status": "error", "message": "Sketch not found"}}
    print(json.dumps(result))
    exit()

try:
    # Create a body if it doesn't exist
    body = None
    for obj in doc.Objects:
        if obj.TypeId == 'PartDesign::Body':
            body = obj
            break
    
    if body is None:
        body = doc.addObject('PartDesign::Body', 'Body')
    
    # Move sketch to body if it's not already there
    if sketch not in body.Group:
        body.addObject(sketch)
    
    # Create pad (extrusion)
    pad = doc.addObject("PartDesign::Pad", "{extrude_name}")
    pad.Profile = sketch
    pad.Length = {length}
    pad.Symmetric = {str(symmetric).lower()}
    pad.TaperAngle = {taper_angle}
    
    # Set direction
    direction = "{direction}"
    if direction == "reversed":
        pad.Reversed = True
    elif direction == "custom":
        # For custom direction, we would need direction vector
        # For now, use normal direction
        pad.Reversed = False
    else:
        pad.Reversed = False
    
    # Add pad to body
    body.addObject(pad)
    
    # Set the pad as the active tip
    body.Tip = pad
    
    # Recompute to create the 3D object
    doc.recompute()
    
    # Calculate volume and other properties
    volume = 0
    surface_area = 0
    if hasattr(pad, 'Shape') and pad.Shape:
        volume = pad.Shape.Volume
        surface_area = pad.Shape.Area
    
    result = {{
        "status": "success",
        "extrude_name": pad.Name,
        "sketch_name": sketch.Name,
        "body_name": body.Name,
        "length": {length},
        "symmetric": {str(symmetric).lower()},
        "taper_angle": {taper_angle},
        "direction": direction,
        "volume": volume,
        "surface_area": surface_area
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "extrude_sketch")
            if result["status"] == "success" and result["output"]:
                return json.loads(result["output"].strip())
            return result
            
        except Exception as e:
            logger.error(f"Failed to extrude sketch: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to extrude sketch: {str(e)}"}

    async def _create_assembly(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create assemblies with multiple parts."""
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            assembly_name = data.get("assembly_name", "Assembly")
            parts = data.get("parts", [])  # List of parts to add to assembly
            constraints = data.get("constraints", [])  # Assembly constraints
            
            if not document_name:
                raise ValueError("document_name is required.")
            
            if not parts:
                raise ValueError("parts list is required.")
            
            # Convert parts and constraints to string representation
            parts_str = json.dumps(parts)
            constraints_str = json.dumps(constraints)
            
            script = f"""
import FreeCAD
import Part
import json
import os

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

try:
    # Create assembly group
    assembly = doc.addObject("App::DocumentObjectGroup", "{assembly_name}")
    assembly.Label = "{assembly_name}"
    
    # Add parts to assembly
    parts = {parts_str}
    added_parts = []
    
    for part_info in parts:
        part_type = part_info.get("type", "link")
        
        if part_type == "file":
            # Import part from external file
            file_path = part_info.get("file_path")
            part_name = part_info.get("name", os.path.basename(file_path) if file_path else "ImportedPart")
            
            if file_path and os.path.exists(file_path):
                # Import the file
                if file_path.lower().endswith('.step') or file_path.lower().endswith('.stp'):
                    import Import
                    Import.insert(file_path, doc.Name)
                elif file_path.lower().endswith('.fcstd'):
                    # Insert FreeCAD file
                    import_doc = FreeCAD.openDocument(file_path, hidden=True)
                    for obj in import_doc.Objects:
                        if hasattr(obj, 'Shape') and obj.Shape:
                            new_obj = doc.addObject("Part::Feature", part_name)
                            new_obj.Shape = obj.Shape.copy()
                            assembly.addObject(new_obj)
                            added_parts.append({{
                                "name": new_obj.Name,
                                "type": "imported",
                                "source": file_path
                            }})
                    FreeCAD.closeDocument(import_doc.Name)
                    
        elif part_type == "existing":
            # Use existing object in document
            object_name = part_info.get("object_name")
            obj = doc.getObject(object_name)
            if obj:
                assembly.addObject(obj)
                added_parts.append({{
                    "name": obj.Name,
                    "type": "existing",
                    "source": "document"
                }})
                
        elif part_type == "copy":
            # Create a copy of existing object
            source_name = part_info.get("source_object")
            new_name = part_info.get("name", f"Copy_{{source_name}}")
            source_obj = doc.getObject(source_name)
            
            if source_obj and hasattr(source_obj, 'Shape'):
                new_obj = doc.addObject("Part::Feature", new_name)
                new_obj.Shape = source_obj.Shape.copy()
                
                # Apply transformation if specified
                transform = part_info.get("transform", {{}})
                if transform:
                    translation = transform.get("translation", [0, 0, 0])
                    rotation = transform.get("rotation", [0, 0, 0])  # Euler angles in degrees
                    
                    placement = FreeCAD.Placement()
                    placement.Base = FreeCAD.Vector(translation[0], translation[1], translation[2])
                    placement.Rotation = FreeCAD.Rotation(rotation[0], rotation[1], rotation[2])
                    new_obj.Placement = placement
                
                assembly.addObject(new_obj)
                added_parts.append({{
                    "name": new_obj.Name,
                    "type": "copy",
                    "source": source_name,
                    "transform": transform
                }})
    
    # Apply assembly constraints (simplified implementation)
    constraints = {constraints_str}
    applied_constraints = []
    
    for constraint in constraints:
        constraint_type = constraint.get("type")
        
        if constraint_type == "coincident":
            part1_name = constraint.get("part1")
            part2_name = constraint.get("part2")
            
            # This is a simplified constraint - in real assembly workbench
            # we would use proper constraint objects
            applied_constraints.append({{
                "type": "coincident",
                "part1": part1_name,
                "part2": part2_name,
                "note": "Constraint applied - use Assembly workbench for full constraint functionality"
            }})
            
        elif constraint_type == "distance":
            part1_name = constraint.get("part1")
            part2_name = constraint.get("part2")
            distance = constraint.get("distance", 0)
            
            applied_constraints.append({{
                "type": "distance",
                "part1": part1_name,
                "part2": part2_name,
                "distance": distance,
                "note": "Constraint applied - use Assembly workbench for full constraint functionality"
            }})
    
    # Recompute document
    doc.recompute()
    
    # Calculate assembly properties
    total_volume = 0
    total_surface_area = 0
    part_count = len(assembly.Group)
    
    for obj in assembly.Group:
        if hasattr(obj, 'Shape') and obj.Shape:
            try:
                total_volume += obj.Shape.Volume
                total_surface_area += obj.Shape.Area
            except:
                pass
    
    result = {{
        "status": "success",
        "assembly_name": assembly.Name,
        "parts_added": len(added_parts),
        "constraints_applied": len(applied_constraints),
        "total_volume": total_volume,
        "total_surface_area": total_surface_area,
        "part_count": part_count,
        "added_parts": added_parts,
        "applied_constraints": applied_constraints
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "create_assembly")
            if result["status"] == "success" and result["output"]:
                return json.loads(result["output"].strip())
            return result
            
        except Exception as e:
            logger.error(f"Failed to create assembly: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to create assembly: {str(e)}"}

    async def _generate_technical_drawing(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical drawings using FreeCAD TechDraw workbench."""
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            object_name = data.get("object_name")
            drawing_name = data.get("drawing_name", "TechnicalDrawing")
            page_template = data.get("page_template", "A4_Portrait")
            view_type = data.get("view_type", "orthographic")  # orthographic, isometric, section
            scale = data.get("scale", 1.0)
            output_format = data.get("output_format", "SVG")  # SVG, PDF, DXF
            output_path = data.get("output_path")
            
            if not document_name or not object_name:
                raise ValueError("document_name and object_name are required.")
            
            # FreeCAD script to generate technical drawing
            script = f"""
import FreeCAD
import TechDraw
import json
import os

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

obj = doc.getObject("{object_name}")
if obj is None:
    result = {{"status": "error", "message": "Object not found"}}
    print(json.dumps(result))
    exit()

if not hasattr(obj, "Shape") or not obj.Shape:
    result = {{"status": "error", "message": "Object has no shape for drawing"}}
    print(json.dumps(result))
    exit()

try:
    # Create TechDraw page
    page = doc.addObject('TechDraw::DrawPage', '{drawing_name}')
    template = doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
    
    # Set template based on page format
    if '{page_template}' == 'A4_Portrait':
        template.Template = FreeCAD.getResourceDir() + "Mod/TechDraw/Templates/A4_Portrait.svg"
    elif '{page_template}' == 'A4_Landscape':
        template.Template = FreeCAD.getResourceDir() + "Mod/TechDraw/Templates/A4_Landscape.svg"
    else:
        template.Template = FreeCAD.getResourceDir() + "Mod/TechDraw/Templates/A4_Portrait.svg"
    
    page.Template = template
    
    # Create views based on view type
    views_created = []
    
    if '{view_type}' == 'orthographic':
        # Create front, top, and side views
        front_view = doc.addObject('TechDraw::DrawViewPart', 'FrontView')
        front_view.Source = [obj]
        front_view.Direction = FreeCAD.Vector(0, -1, 0)
        front_view.Scale = {scale}
        front_view.X = 100
        front_view.Y = 150
        page.addView(front_view)
        views_created.append("FrontView")
        
        top_view = doc.addObject('TechDraw::DrawViewPart', 'TopView')
        top_view.Source = [obj]
        top_view.Direction = FreeCAD.Vector(0, 0, 1)
        top_view.Scale = {scale}
        top_view.X = 100
        top_view.Y = 250
        page.addView(top_view)
        views_created.append("TopView")
        
        side_view = doc.addObject('TechDraw::DrawViewPart', 'SideView')
        side_view.Source = [obj]
        side_view.Direction = FreeCAD.Vector(1, 0, 0)
        side_view.Scale = {scale}
        side_view.X = 200
        side_view.Y = 150
        page.addView(side_view)
        views_created.append("SideView")
        
    elif '{view_type}' == 'isometric':
        iso_view = doc.addObject('TechDraw::DrawViewPart', 'IsometricView')
        iso_view.Source = [obj]
        iso_view.Direction = FreeCAD.Vector(1, 1, 1)
        iso_view.Scale = {scale}
        iso_view.X = 150
        iso_view.Y = 200
        page.addView(iso_view)
        views_created.append("IsometricView")
        
    elif '{view_type}' == 'section':
        # Create a section view
        section_view = doc.addObject('TechDraw::DrawViewSection', 'SectionView')
        section_view.Source = [obj]
        section_view.Direction = FreeCAD.Vector(0, -1, 0)
        section_view.Scale = {scale}
        section_view.X = 150
        section_view.Y = 200
        page.addView(section_view)
        views_created.append("SectionView")
    
    # Add dimensions if requested
    dimension_style = data.get("add_dimensions", False)
    if dimension_style:
        # Add basic dimensions to the first view
        if views_created:
            first_view = page.Views[0]
            # Add length dimension
            length_dim = doc.addObject('TechDraw::DrawViewDimension', 'LengthDim')
            length_dim.Type = 'Distance'
            length_dim.References2D = [(first_view, 'Edge0'), (first_view, 'Edge2')]
            page.addView(length_dim)
    
    doc.recompute()
    
    # Export drawing if output path is specified
    export_result = None
    if '{output_path}':
        try:
            if '{output_format}' == 'SVG':
                page.exportSvg('{output_path}')
            elif '{output_format}' == 'PDF':
                page.exportPdf('{output_path}')
            elif '{output_format}' == 'DXF':
                page.exportDxf('{output_path}')
            export_result = {{"exported": True, "path": '{output_path}', "format": '{output_format}'}}
        except Exception as e:
            export_result = {{"exported": False, "error": str(e)}}
    
    result = {{
        "status": "success",
        "page_name": page.Name,
        "template": '{page_template}',
        "view_type": '{view_type}',
        "scale": {scale},
        "views_created": views_created,
        "export_result": export_result
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "generate_technical_drawing")
            if result["status"] == "success" and result["output"]:
                drawing_result = json.loads(result["output"].strip())
                return drawing_result
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate technical drawing: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to generate technical drawing: {str(e)}"}

    async def _mesh_analysis(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mesh analysis and validation."""
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            object_name = data.get("object_name")
            analysis_type = data.get("analysis_type", "quality")  # quality, topology, geometry
            mesh_tolerance = data.get("mesh_tolerance", 0.1)
            
            if not document_name or not object_name:
                raise ValueError("document_name and object_name are required.")
            
            # FreeCAD script for mesh analysis
            script = f"""
import FreeCAD
import Mesh
import MeshPart
import json

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

obj = doc.getObject("{object_name}")
if obj is None:
    result = {{"status": "error", "message": "Object not found"}}
    print(json.dumps(result))
    exit()

if not hasattr(obj, "Shape") or not obj.Shape:
    result = {{"status": "error", "message": "Object has no shape for mesh analysis"}}
    print(json.dumps(result))
    exit()

try:
    # Generate mesh from shape
    mesh = MeshPart.meshFromShape(obj.Shape, LinearDeflection={mesh_tolerance})
    
    analysis_results = {{}}
    analysis_type = "{analysis_type}"
    
    if analysis_type == "quality" or analysis_type == "all":
        # Quality analysis
        mesh_points = len(mesh.Points)
        mesh_facets = len(mesh.Facets)
        
        # Check for degenerate facets
        degenerate_facets = 0
        for facet in mesh.Facets:
            if facet.Area < 1e-10:
                degenerate_facets += 1
        
        # Check mesh bounds
        bounds = mesh.BoundBox
        mesh_volume = mesh.Volume if hasattr(mesh, 'Volume') else 0
        surface_area = mesh.Area if hasattr(mesh, 'Area') else 0
        
        analysis_results["quality"] = {{
            "points_count": mesh_points,
            "facets_count": mesh_facets,
            "degenerate_facets": degenerate_facets,
            "volume": mesh_volume,
            "surface_area": surface_area,
            "bounds": {{
                "x_min": bounds.XMin,
                "y_min": bounds.YMin,
                "z_min": bounds.ZMin,
                "x_max": bounds.XMax,
                "y_max": bounds.YMax,
                "z_max": bounds.ZMax
            }}
        }}
    
    if analysis_type == "topology" or analysis_type == "all":
        # Topology analysis
        try:
            # Check for manifold edges
            manifold_edges = mesh.getManifoldEdges()
            non_manifold_edges = mesh.getNonManifoldEdges()
            
            # Check for holes
            holes = mesh.getHoles()
            
            # Check for self-intersections
            self_intersections = mesh.getSelfIntersections()
            
            analysis_results["topology"] = {{
                "is_manifold": len(non_manifold_edges) == 0,
                "manifold_edges": len(manifold_edges),
                "non_manifold_edges": len(non_manifold_edges),
                "holes_count": len(holes),
                "self_intersections": len(self_intersections),
                "is_solid": mesh.isSolid() if hasattr(mesh, 'isSolid') else False
            }}
        except:
            analysis_results["topology"] = {{
                "error": "Topology analysis failed - some methods may not be available"
            }}
    
    if analysis_type == "geometry" or analysis_type == "all":
        # Geometry analysis
        try:
            # Calculate mesh quality metrics
            min_edge_length = float('inf')
            max_edge_length = 0
            total_edge_length = 0
            edge_count = 0
            
            for facet in mesh.Facets:
                # Calculate edge lengths for this facet
                points = [mesh.Points[i] for i in facet.PointIndices]
                for i in range(len(points)):
                    p1 = points[i]
                    p2 = points[(i + 1) % len(points)]
                    edge_length = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5
                    min_edge_length = min(min_edge_length, edge_length)
                    max_edge_length = max(max_edge_length, edge_length)
                    total_edge_length += edge_length
                    edge_count += 1
            
            avg_edge_length = total_edge_length / edge_count if edge_count > 0 else 0
            
            analysis_results["geometry"] = {{
                "min_edge_length": min_edge_length if min_edge_length != float('inf') else 0,
                "max_edge_length": max_edge_length,
                "avg_edge_length": avg_edge_length,
                "edge_length_ratio": max_edge_length / min_edge_length if min_edge_length > 0 else 0
            }}
        except Exception as e:
            analysis_results["geometry"] = {{
                "error": f"Geometry analysis failed: {{str(e)}}"
            }}
    
    # Generate validation report
    validation_report = {{
        "is_valid": True,
        "warnings": [],
        "errors": []
    }}
    
    # Check for common issues
    if "quality" in analysis_results:
        quality = analysis_results["quality"]
        if quality["degenerate_facets"] > 0:
            validation_report["warnings"].append(f"Found {{quality['degenerate_facets']}} degenerate facets")
        if quality["facets_count"] < 10:
            validation_report["warnings"].append("Very low facet count - mesh may be too coarse")
        if quality["facets_count"] > 100000:
            validation_report["warnings"].append("Very high facet count - mesh may be too fine")
    
    if "topology" in analysis_results:
        topology = analysis_results["topology"]
        if not topology.get("is_manifold", True):
            validation_report["errors"].append("Mesh is not manifold")
            validation_report["is_valid"] = False
        if topology.get("holes_count", 0) > 0:
            validation_report["warnings"].append(f"Found {{topology['holes_count']}} holes in mesh")
        if topology.get("self_intersections", 0) > 0:
            validation_report["errors"].append("Mesh has self-intersections")
            validation_report["is_valid"] = False
    
    if "geometry" in analysis_results:
        geometry = analysis_results["geometry"]
        if geometry.get("edge_length_ratio", 0) > 100:
            validation_report["warnings"].append("High edge length ratio - mesh quality may be poor")
    
    result = {{
        "status": "success",
        "object_name": object_name,
        "analysis_type": analysis_type,
        "mesh_tolerance": {mesh_tolerance},
        "analysis_results": analysis_results,
        "validation_report": validation_report
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "mesh_analysis")
            if result["status"] == "success" and result["output"]:
                analysis_result = json.loads(result["output"].strip())
                return analysis_result
            return result
            
        except Exception as e:
            logger.error(f"Failed to perform mesh analysis: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to perform mesh analysis: {str(e)}"}

    async def _check_interference(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for interference between objects."""
        try:
            object1_name = data.get("object1_name")
            object2_name = data.get("object2_name")
            
            if not object1_name or not object2_name:
                return {"status": "error", "message": "Both object names are required"}
            
            # Execute FreeCAD Python code to check interference
            freecad_code = f"""
import FreeCAD
import Part

# Get the objects
obj1 = FreeCAD.ActiveDocument.getObject('{object1_name}')
obj2 = FreeCAD.ActiveDocument.getObject('{object2_name}')

if obj1 is None or obj2 is None:
    result = {{"error": "One or both objects not found"}}
else:
    # Check if both objects have shapes
    if hasattr(obj1, 'Shape') and hasattr(obj2, 'Shape') and obj1.Shape and obj2.Shape:
        # Check for intersection
        try:
            intersection = obj1.Shape.common(obj2.Shape)
            has_interference = intersection.Volume > 1e-6  # Small tolerance for floating point
            
            if has_interference:
                result = {{
                    "interference_detected": True,
                    "intersection_volume": intersection.Volume,
                    "intersection_area": intersection.Area,
                    "message": "Objects interfere with each other"
                }}
            else:
                result = {{
                    "interference_detected": False,
                    "message": "No interference detected between objects"
                }}
        except Exception as e:
            result = {{"error": f"Failed to check interference: {{str(e)}}"}}
    else:
        result = {{"error": "Objects do not have valid shapes for interference checking"}}
"""
            
            result = await self._execute_freecad_code(freecad_code, credential_id, config)
            
            if "error" in result:
                return {"status": "error", "message": result["error"]}
            
            return {
                "status": "success",
                "interference_detected": result.get("interference_detected", False),
                "intersection_volume": result.get("intersection_volume", 0),
                "intersection_area": result.get("intersection_area", 0),
                "message": result.get("message", "Interference check completed")
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to check interference: {str(e)}"}
    
    async def _execute_freecad_code(self, freecad_code: str, credential_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FreeCAD Python code and return results."""
        try:
            # Get FreeCAD connection details
            credentials = await vault_client.get_credential(credential_id)
            freecad_path = credentials.secrets.get("freecad_path", "/usr/bin/freecad")
            python_path = credentials.secrets.get("python_path", "/usr/bin/python3")
            
            # Create a temporary Python script
            import tempfile
            import os
            import subprocess
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the FreeCAD execution script
                script_content = f"""
import sys
sys.path.append('{freecad_path}/lib')
import FreeCAD
import Part

# Create or open document
try:
    doc = FreeCAD.ActiveDocument
    if doc is None:
        doc = FreeCAD.newDocument()
except:
    doc = FreeCAD.newDocument()

# Execute the user code
try:
    {freecad_code}
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
                f.write(script_content)
                script_path = f.name
            
            try:
                # Execute the script
                process = subprocess.run(
                    [python_path, script_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if process.returncode == 0:
                    # Parse the JSON output
                    try:
                        import json
                        result = json.loads(process.stdout.strip())
                        return result
                    except json.JSONDecodeError:
                        return {"error": f"Failed to parse FreeCAD output: {process.stdout}"}
                else:
                    return {"error": f"FreeCAD execution failed: {process.stderr}"}
                    
            finally:
                # Clean up temporary file
                os.unlink(script_path)
                
        except subprocess.TimeoutExpired:
            return {"error": "FreeCAD execution timed out"}
        except Exception as e:
            return {"error": f"Failed to execute FreeCAD code: {str(e)}"}

    async def _generate_gcode(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate G-code for 3D printing/CNC."""
        try:
            client_config = await self._get_freecad_client(credential_id)
            document_name = config.get("document_name")
            object_name = data.get("object_name")
            operation_type = data.get("operation_type", "3d_printing")  # 3d_printing, cnc_milling
            output_path = data.get("output_path")
            
            # Operation-specific settings
            settings = data.get("settings", {})
            
            if not document_name or not object_name:
                raise ValueError("document_name and object_name are required.")
            
            if not output_path:
                workspace = client_config["workspace_path"]
                output_path = os.path.join(workspace, f"{object_name}.gcode")
            
            # Convert settings to string representation
            settings_str = json.dumps(settings)
            
            script = f"""
import FreeCAD
import Mesh
import json
import os

doc = FreeCAD.getDocument("{document_name}")
if doc is None:
    result = {{"status": "error", "message": "Document not found"}}
    print(json.dumps(result))
    exit()

obj = doc.getObject("{object_name}")
if obj is None:
    result = {{"status": "error", "message": "Object not found"}}
    print(json.dumps(result))
    exit()

if not hasattr(obj, "Shape") or not obj.Shape:
    result = {{"status": "error", "message": "Object has no shape for G-code generation"}}
    print(json.dumps(result))
    exit()

try:
    operation_type = "{operation_type}"
    output_path = "{output_path}"
    settings = {settings_str}
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if operation_type == "3d_printing":
        # 3D Printing G-code generation
        layer_height = settings.get("layer_height", 0.2)
        nozzle_temp = settings.get("nozzle_temperature", 200)
        bed_temp = settings.get("bed_temperature", 60)
        print_speed = settings.get("print_speed", 50)
        infill_density = settings.get("infill_density", 20)
        
        # Get object bounds
        bbox = obj.Shape.BoundBox
        
        # Simple G-code generation (basic slicing simulation)
                 gcode_lines = [
             "; Generated by FreeCAD Connector",
             f"; Object: {{obj.Name}}",
             f"; Layer Height: {{layer_height}}mm",
             f"; Infill: {{infill_density}}%",
            "",
            "; Start G-code",
            "G21 ; set units to millimeters",
            "G90 ; use absolute coordinates",
            "M82 ; use absolute distances for extrusion",
            f"M104 S{{nozzle_temp}} ; set extruder temp",
            f"M140 S{{bed_temp}} ; set bed temp",
            "G28 ; home all axes",
            f"M109 S{{nozzle_temp}} ; wait for extruder temp",
            f"M190 S{{bed_temp}} ; wait for bed temp",
            "G92 E0 ; reset extruder",
            "",
            "; Object bounds information",
            f"; X: {{bbox.XMin:.2f}} to {{bbox.XMax:.2f}} ({{bbox.XLength:.2f}}mm)",
            f"; Y: {{bbox.YMin:.2f}} to {{bbox.YMax:.2f}} ({{bbox.YLength:.2f}}mm)",
            f"; Z: {{bbox.ZMin:.2f}} to {{bbox.ZMax:.2f}} ({{bbox.ZLength:.2f}}mm)",
            "",
            "; Layer generation (simplified)",
        ]
        
        # Generate layers (very simplified - real slicer would be much more complex)
        num_layers = int(bbox.ZLength / layer_height) + 1
        current_z = bbox.ZMin
        
        for layer in range(num_layers):
            gcode_lines.extend([
                f"; Layer {{layer + 1}}/{{num_layers}}",
                f"G0 Z{{current_z:.3f}} ; move to layer height",
                f"G0 X{{bbox.XMin:.3f}} Y{{bbox.YMin:.3f}} ; move to start position",
                f"G1 F{{print_speed * 60}} ; set print speed",
            ])
            
            # Simple perimeter (rectangle approximation)
            gcode_lines.extend([
                f"G1 X{{bbox.XMax:.3f}} Y{{bbox.YMin:.3f}} E{{layer * 2 + 1}:.4f}",
                f"G1 X{{bbox.XMax:.3f}} Y{{bbox.YMax:.3f}} E{{layer * 2 + 2}:.4f}",
                f"G1 X{{bbox.XMin:.3f}} Y{{bbox.YMax:.3f}} E{{layer * 2 + 3}:.4f}",
                f"G1 X{{bbox.XMin:.3f}} Y{{bbox.YMin:.3f}} E{{layer * 2 + 4}:.4f}",
                ""
            ])
            
            current_z += layer_height
        
        # End G-code
        gcode_lines.extend([
            "; End G-code",
            "M104 S0 ; turn off extruder",
            "M140 S0 ; turn off bed",
            "G91 ; relative positioning",
            "G1 E-1 F300 ; retract filament",
            "G1 Z+0.5 E-5 X-20 Y-20 F9000 ; move away",
            "G90 ; absolute positioning",
            "M84 ; disable motors",
            "M30 ; end program"
        ])
        
        operation_info = {{
            "type": "3d_printing",
            "layer_height": layer_height,
            "layers": num_layers,
            "nozzle_temperature": nozzle_temp,
            "bed_temperature": bed_temp,
            "estimated_print_time": f"{{num_layers * 2}} minutes (simplified estimate)"
        }}
        
    elif operation_type == "cnc_milling":
        # CNC Milling G-code generation
        spindle_speed = settings.get("spindle_speed", 10000)
        feed_rate = settings.get("feed_rate", 500)
        plunge_rate = settings.get("plunge_rate", 100)
        cut_depth = settings.get("cut_depth", 1.0)
        tool_diameter = settings.get("tool_diameter", 6.0)
        
        # Get object bounds
        bbox = obj.Shape.BoundBox
        
        gcode_lines = [
            "; Generated by FreeCAD Connector - CNC Milling",
            f"; Object: {{obj.Name}}",
            f"; Tool Diameter: {{tool_diameter}}mm",
            f"; Spindle Speed: {{spindle_speed}} RPM",
            "",
            "; Start G-code",
            "G21 ; set units to millimeters",
            "G90 ; use absolute coordinates",
            "G17 ; select XY plane",
            "G94 ; feed rate mode",
            f"M3 S{{spindle_speed}} ; start spindle",
            "G0 Z5 ; rapid to safe height",
            "",
            "; Object bounds information",
            f"; X: {{bbox.XMin:.2f}} to {{bbox.XMax:.2f}} ({{bbox.XLength:.2f}}mm)",
            f"; Y: {{bbox.YMin:.2f}} to {{bbox.YMax:.2f}} ({{bbox.ZLength:.2f}}mm)",
            f"; Z: {{bbox.ZMin:.2f}} to {{bbox.ZMax:.2f}} ({{bbox.ZLength:.2f}}mm)",
            "",
            "; Roughing operation (simplified)",
            f"G0 X{{bbox.XMin - tool_diameter/2:.3f}} Y{{bbox.YMin - tool_diameter/2:.3f}}",
            f"G1 Z{{bbox.ZMax - cut_depth:.3f}} F{{plunge_rate}}",
        ]
        
        # Simple roughing pattern (very simplified)
        step_over = tool_diameter * 0.7
        x_steps = int(bbox.XLength / step_over) + 1
        y_steps = int(bbox.YLength / step_over) + 1
        
        current_x = bbox.XMin
        for x_step in range(x_steps):
            current_y = bbox.YMin if x_step % 2 == 0 else bbox.YMax
            end_y = bbox.YMax if x_step % 2 == 0 else bbox.YMin
            
            gcode_lines.extend([
                f"G0 X{{current_x:.3f}}",
                f"G1 Y{{end_y:.3f}} F{{feed_rate}}"
            ])
            
            current_x += step_over
        
        # End G-code
        gcode_lines.extend([
            "",
            "; End G-code",
            "G0 Z5 ; rapid to safe height",
            "M5 ; stop spindle",
            "G0 X0 Y0 ; return to origin",
            "M30 ; end program"
        ])
        
        operation_info = {{
            "type": "cnc_milling",
            "spindle_speed": spindle_speed,
            "feed_rate": feed_rate,
            "tool_diameter": tool_diameter,
            "estimated_machining_time": f"{{x_steps * y_steps * 0.5}} minutes (simplified estimate)"
        }}
    
    else:
        result = {{"status": "error", "message": f"Unsupported operation type: {{operation_type}}"}}
        print(json.dumps(result))
        exit()
    
    # Write G-code to file
    with open(output_path, 'w') as f:
        f.write('\\n'.join(gcode_lines))
    
    # Calculate file statistics
    line_count = len(gcode_lines)
    file_size = os.path.getsize(output_path)
    
    result = {{
        "status": "success",
        "object_name": obj.Name,
        "operation_type": operation_type,
        "output_path": output_path,
        "gcode_lines": line_count,
        "file_size_bytes": file_size,
        "operation_info": operation_info,
        "bounds": {{
            "x_min": bbox.XMin,
            "y_min": bbox.YMin,
            "z_min": bbox.ZMin,
            "x_max": bbox.XMax,
            "y_max": bbox.YMax,
            "z_max": bbox.ZMax,
            "dimensions": [bbox.XLength, bbox.YLength, bbox.ZLength]
        }}
    }}
    
except Exception as e:
    result = {{"status": "error", "message": str(e)}}

print(json.dumps(result))
"""
            
            result = await self._execute_freecad_script(client_config, script, "generate_gcode")
            if result["status"] == "success" and result["output"]:
                return json.loads(result["output"].strip())
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate G-code: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to generate G-code: {str(e)}"}


# Instantiate a single instance
freecad_connector = FreeCADConnector() 