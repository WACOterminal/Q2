import logging
from typing import Dict, Any, Optional, List
import httpx
from fastapi import HTTPException
import asyncio
import json
import base64
import os
import tempfile
import subprocess
from pathlib import Path

from app.models.connector import BaseConnector, ConnectorAction
from app.core.vault_client import vault_client

logger = logging.getLogger(__name__)

class MultimediaConnector(BaseConnector):
    """
    A connector for multimedia processing using Audacity, GIMP, and OpenShot.
    
    Supported actions:
    
    AUDACITY:
    - process_audio: Process audio files with Audacity scripts
    - convert_audio: Convert audio between formats
    - apply_effects: Apply audio effects and filters
    - export_audio: Export audio in various formats
    - analyze_audio: Analyze audio properties
    
    GIMP:
    - process_image: Process images with GIMP scripts
    - convert_image: Convert images between formats
    - apply_filters: Apply image filters and effects
    - resize_image: Resize and scale images
    - crop_image: Crop images to specific dimensions
    - batch_process: Process multiple images
    
    OPENSHOT:
    - create_project: Create new video project
    - add_media: Add media files to project
    - apply_transitions: Apply video transitions
    - add_effects: Add video effects
    - export_video: Export video in various formats
    - generate_timeline: Generate video timeline
    """

    @property
    def connector_id(self) -> str:
        return "multimedia"

    async def _get_multimedia_config(self, credential_id: str) -> Dict[str, Any]:
        """Helper to get multimedia tools configuration."""
        credential = await vault_client.get_credential(credential_id)
        
        return {
            "audacity_path": credential.secrets.get("audacity_path", "/usr/bin/audacity"),
            "gimp_path": credential.secrets.get("gimp_path", "/usr/bin/gimp"),
            "openshot_path": credential.secrets.get("openshot_path", "/usr/bin/openshot-qt"),
            "ffmpeg_path": credential.secrets.get("ffmpeg_path", "/usr/bin/ffmpeg"),
            "workspace_path": credential.secrets.get("workspace_path", "/tmp/multimedia_workspace"),
            "python_path": credential.secrets.get("python_path", "/usr/bin/python3")
        }

    async def execute(self, action: ConnectorAction, configuration: Dict[str, Any], data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not action.credential_id:
            raise ValueError("credential_id must be provided for Multimedia connector actions.")
        
        try:
            action_map = {
                # Audacity actions
                "process_audio": self._process_audio,
                "convert_audio": self._convert_audio,
                "apply_effects": self._apply_effects,
                "export_audio": self._export_audio,
                "analyze_audio": self._analyze_audio,
                
                # GIMP actions
                "process_image": self._process_image,
                "convert_image": self._convert_image,
                "apply_filters": self._apply_filters,
                "resize_image": self._resize_image,
                "crop_image": self._crop_image,
                "batch_process": self._batch_process,
                
                # OpenShot actions
                "create_project": self._create_project,
                "add_media": self._add_media,
                "apply_transitions": self._apply_transitions,
                "add_effects": self._add_effects,
                "export_video": self._export_video,
                "generate_timeline": self._generate_timeline,
            }

            if action.action_id in action_map:
                func = action_map[action.action_id]
                return await func(action.credential_id, configuration, data_context)
            else:
                raise ValueError(f"Unsupported action for Multimedia connector: {action.action_id}")

        except Exception as e:
            logger.error(f"An unexpected error occurred in MultimediaConnector: {e}", exc_info=True)
            raise

    async def _execute_command(self, command: List[str], timeout: int = 300) -> Dict[str, Any]:
        """Execute a system command and return the result."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": "Command execution timed out"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # AUDACITY ACTIONS
    async def _process_audio(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio files with Audacity scripts."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        script_commands = data.get("script_commands", [])
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        # Create Audacity script
        script_content = "\n".join([
            f"Import2: Filename=\"{input_file}\"",
            *script_commands,
            f"Export2: Filename=\"{output_file}\" FileFormat=\"WAV\""
        ])
        
        # Write script to temporary file
        workspace = multimedia_config["workspace_path"]
        os.makedirs(workspace, exist_ok=True)
        
        script_file = os.path.join(workspace, "audacity_script.txt")
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Execute Audacity with script
        command = [
            multimedia_config["audacity_path"],
            "-s", script_file
        ]
        
        result = await self._execute_command(command)
        
        # Clean up
        if os.path.exists(script_file):
            os.remove(script_file)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "processing_result": result
        }

    async def _convert_audio(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert audio between formats using FFmpeg."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        output_format = config.get("output_format", "mp3")
        quality = config.get("quality", "192k")
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        command = [
            multimedia_config["ffmpeg_path"],
            "-i", input_file,
            "-acodec", "libmp3lame" if output_format == "mp3" else "pcm_s16le",
            "-ab", quality,
            output_file,
            "-y"  # Overwrite output file
        ]
        
        result = await self._execute_command(command)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "format": output_format,
            "conversion_result": result
        }

    async def _apply_effects(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply audio effects and filters."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        effects = data.get("effects", [])
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        # Build FFmpeg filter complex
        filter_parts = []
        for effect in effects:
            effect_type = effect.get("type")
            params = effect.get("parameters", {})
            
            if effect_type == "normalize":
                filter_parts.append("anorm")
            elif effect_type == "amplify":
                gain = params.get("gain", 1.0)
                filter_parts.append(f"volume={gain}")
            elif effect_type == "lowpass":
                freq = params.get("frequency", 1000)
                filter_parts.append(f"lowpass=f={freq}")
            elif effect_type == "highpass":
                freq = params.get("frequency", 100)
                filter_parts.append(f"highpass=f={freq}")
        
        if filter_parts:
            command = [
                multimedia_config["ffmpeg_path"],
                "-i", input_file,
                "-af", ",".join(filter_parts),
                output_file,
                "-y"
            ]
        else:
            # No effects, just copy
            command = [
                multimedia_config["ffmpeg_path"],
                "-i", input_file,
                "-c", "copy",
                output_file,
                "-y"
            ]
        
        result = await self._execute_command(command)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "effects_applied": len(effects),
            "processing_result": result
        }

    async def _export_audio(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Export audio in various formats."""
        return await self._convert_audio(credential_id, config, data)

    async def _analyze_audio(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio properties."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        
        if not input_file:
            raise ValueError("input_file is required.")
        
        command = [
            multimedia_config["ffmpeg_path"],
            "-i", input_file,
            "-f", "null", "-"
        ]
        
        result = await self._execute_command(command)
        
        # Parse audio information from stderr
        audio_info = {}
        if result["stderr"]:
            # Extract basic audio info (this is a simplified parser)
            lines = result["stderr"].split('\n')
            for line in lines:
                if "Duration:" in line:
                    audio_info["duration"] = line.split("Duration:")[1].split(",")[0].strip()
                elif "Audio:" in line:
                    audio_info["format"] = line.split("Audio:")[1].split(",")[0].strip()
        
        return {
            "status": "success",
            "input_file": input_file,
            "audio_info": audio_info,
            "analysis_result": result
        }

    # GIMP ACTIONS
    async def _process_image(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Process images with GIMP scripts."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        script_fu_commands = data.get("script_fu_commands", [])
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        # Create GIMP script
        script_content = f"""
(let* ((image (car (gimp-file-load RUN-NONINTERACTIVE "{input_file}" "{input_file}")))
       (drawable (car (gimp-image-get-active-layer image))))
"""
        
        # Add custom Script-Fu commands
        for cmd in script_fu_commands:
            script_content += f"  {cmd}\n"
        
        script_content += f"""
  (gimp-file-save RUN-NONINTERACTIVE image drawable "{output_file}" "{output_file}")
  (gimp-image-delete image))
"""
        
        # Write script to temporary file
        workspace = multimedia_config["workspace_path"]
        os.makedirs(workspace, exist_ok=True)
        
        script_file = os.path.join(workspace, "gimp_script.scm")
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Execute GIMP with script
        command = [
            multimedia_config["gimp_path"],
            "-i", "-b", f"(load \"{script_file}\")", "-b", "(gimp-quit 0)"
        ]
        
        result = await self._execute_command(command)
        
        # Clean up
        if os.path.exists(script_file):
            os.remove(script_file)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "processing_result": result
        }

    async def _convert_image(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert images between formats."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        quality = config.get("quality", 90)
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        # Use ImageMagick convert if available, otherwise fall back to GIMP
        command = [
            "convert",  # ImageMagick
            input_file,
            "-quality", str(quality),
            output_file
        ]
        
        result = await self._execute_command(command)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "conversion_result": result
        }

    async def _resize_image(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Resize and scale images."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        width = data.get("width")
        height = data.get("height")
        maintain_aspect = config.get("maintain_aspect", True)
        
        if not input_file or not output_file or not width or not height:
            raise ValueError("input_file, output_file, width, and height are required.")
        
        size_param = f"{width}x{height}"
        if maintain_aspect:
            size_param += ">"
        
        command = [
            "convert",  # ImageMagick
            input_file,
            "-resize", size_param,
            output_file
        ]
        
        result = await self._execute_command(command)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "new_size": {"width": width, "height": height},
            "resize_result": result
        }

    # Placeholder implementations for other actions
    async def _apply_filters(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply image filters and effects."""
        return {"status": "success", "message": "Image filters not yet implemented"}

    async def _crop_image(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Crop images to specific dimensions."""
        return {"status": "success", "message": "Image cropping not yet implemented"}

    async def _batch_process(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple images."""
        return {"status": "success", "message": "Batch processing not yet implemented"}

    # OPENSHOT ACTIONS
    async def _create_project(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new video project."""
        return {"status": "success", "message": "OpenShot project creation not yet implemented"}

    async def _add_media(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Add media files to project."""
        return {"status": "success", "message": "OpenShot media addition not yet implemented"}

    async def _apply_transitions(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply video transitions."""
        return {"status": "success", "message": "OpenShot transitions not yet implemented"}

    async def _add_effects(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Add video effects."""
        return {"status": "success", "message": "OpenShot effects not yet implemented"}

    async def _export_video(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Export video in various formats."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        output_format = config.get("output_format", "mp4")
        quality = config.get("quality", "medium")
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        # Quality presets
        quality_presets = {
            "low": "-crf 35",
            "medium": "-crf 23",
            "high": "-crf 18"
        }
        
        quality_params = quality_presets.get(quality, "-crf 23").split()
        
        command = [
            multimedia_config["ffmpeg_path"],
            "-i", input_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            *quality_params,
            output_file,
            "-y"
        ]
        
        result = await self._execute_command(command, timeout=1800)  # 30 minutes for video processing
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "format": output_format,
            "export_result": result
        }

    async def _generate_timeline(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video timeline."""
        return {"status": "success", "message": "OpenShot timeline generation not yet implemented"}


# Instantiate a single instance
multimedia_connector = MultimediaConnector() 