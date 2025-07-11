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
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        filters = data.get("filters", [])
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        # Build ImageMagick command with filters
        command = ["convert", input_file]
        
        applied_filters = []
        for filter_config in filters:
            filter_type = filter_config.get("type")
            params = filter_config.get("parameters", {})
            
            if filter_type == "blur":
                radius = params.get("radius", 1.0)
                sigma = params.get("sigma", 1.0)
                command.extend(["-blur", f"{radius}x{sigma}"])
                applied_filters.append(f"blur(radius={radius}, sigma={sigma})")
                
            elif filter_type == "sharpen":
                radius = params.get("radius", 1.0)
                sigma = params.get("sigma", 1.0)
                command.extend(["-sharpen", f"{radius}x{sigma}"])
                applied_filters.append(f"sharpen(radius={radius}, sigma={sigma})")
                
            elif filter_type == "emboss":
                radius = params.get("radius", 1.0)
                sigma = params.get("sigma", 1.0)
                command.extend(["-emboss", f"{radius}x{sigma}"])
                applied_filters.append(f"emboss(radius={radius}, sigma={sigma})")
                
            elif filter_type == "edge":
                radius = params.get("radius", 1.0)
                command.extend(["-edge", str(radius)])
                applied_filters.append(f"edge(radius={radius})")
                
            elif filter_type == "normalize":
                command.extend(["-normalize"])
                applied_filters.append("normalize")
                
            elif filter_type == "equalize":
                command.extend(["-equalize"])
                applied_filters.append("equalize")
                
            elif filter_type == "contrast":
                level = params.get("level", 0)
                command.extend(["-contrast-stretch", f"{level}%"])
                applied_filters.append(f"contrast(level={level}%)")
                
            elif filter_type == "brightness":
                level = params.get("level", 0)
                command.extend(["-brightness-contrast", f"{level}x0"])
                applied_filters.append(f"brightness(level={level})")
                
            elif filter_type == "gamma":
                value = params.get("value", 1.0)
                command.extend(["-gamma", str(value)])
                applied_filters.append(f"gamma(value={value})")
                
            elif filter_type == "sepia":
                threshold = params.get("threshold", "80%")
                command.extend(["-sepia-tone", threshold])
                applied_filters.append(f"sepia(threshold={threshold})")
                
            elif filter_type == "grayscale":
                command.extend(["-colorspace", "Gray"])
                applied_filters.append("grayscale")
                
            elif filter_type == "negate":
                command.extend(["-negate"])
                applied_filters.append("negate")
                
            elif filter_type == "solarize":
                threshold = params.get("threshold", "50%")
                command.extend(["-solarize", threshold])
                applied_filters.append(f"solarize(threshold={threshold})")
                
            elif filter_type == "oil_paint":
                radius = params.get("radius", 3.0)
                command.extend(["-paint", str(radius)])
                applied_filters.append(f"oil_paint(radius={radius})")
                
            elif filter_type == "charcoal":
                radius = params.get("radius", 1.0)
                sigma = params.get("sigma", 1.0)
                command.extend(["-charcoal", f"{radius}x{sigma}"])
                applied_filters.append(f"charcoal(radius={radius}, sigma={sigma})")
                
            elif filter_type == "posterize":
                levels = params.get("levels", 8)
                command.extend(["-posterize", str(levels)])
                applied_filters.append(f"posterize(levels={levels})")
                
            elif filter_type == "noise":
                noise_type = params.get("noise_type", "gaussian")
                command.extend(["+noise", noise_type])
                applied_filters.append(f"noise(type={noise_type})")
                
            elif filter_type == "despeckle":
                command.extend(["-despeckle"])
                applied_filters.append("despeckle")
                
            elif filter_type == "median":
                radius = params.get("radius", 1.0)
                command.extend(["-median", str(radius)])
                applied_filters.append(f"median(radius={radius})")
        
        # Add output file
        command.append(output_file)
        
        result = await self._execute_command(command)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "applied_filters": applied_filters,
            "filters_count": len(applied_filters),
            "processing_result": result
        }

    async def _crop_image(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Crop images to specific dimensions."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_file = config.get("input_file")
        output_file = data.get("output_file")
        
        # Crop parameters
        x = data.get("x", 0)
        y = data.get("y", 0)
        width = data.get("width")
        height = data.get("height")
        
        # Alternative: crop by percentage
        crop_left = data.get("crop_left")  # percentage
        crop_top = data.get("crop_top")    # percentage
        crop_right = data.get("crop_right")  # percentage
        crop_bottom = data.get("crop_bottom")  # percentage
        
        if not input_file or not output_file:
            raise ValueError("input_file and output_file are required.")
        
        command = ["convert", input_file]
        
        if width and height:
            # Crop by pixel dimensions
            crop_geometry = f"{width}x{height}+{x}+{y}"
            command.extend(["-crop", crop_geometry])
            crop_info = f"crop to {width}x{height} at ({x}, {y})"
        elif crop_left is not None or crop_top is not None or crop_right is not None or crop_bottom is not None:
            # Crop by percentage
            # First get image dimensions
            identify_cmd = ["identify", "-format", "%wx%h", input_file]
            identify_result = await self._execute_command(identify_cmd)
            
            if identify_result["status"] == "success":
                dimensions = identify_result["stdout"].strip()
                img_width, img_height = map(int, dimensions.split('x'))
                
                # Calculate pixel values from percentages
                left_pixels = int((crop_left or 0) * img_width / 100)
                top_pixels = int((crop_top or 0) * img_height / 100)
                right_pixels = int((crop_right or 0) * img_width / 100)
                bottom_pixels = int((crop_bottom or 0) * img_height / 100)
                
                # Calculate new dimensions
                new_width = img_width - left_pixels - right_pixels
                new_height = img_height - top_pixels - bottom_pixels
                
                crop_geometry = f"{new_width}x{new_height}+{left_pixels}+{top_pixels}"
                command.extend(["-crop", crop_geometry])
                crop_info = f"crop by percentage: left={crop_left}%, top={crop_top}%, right={crop_right}%, bottom={crop_bottom}%"
            else:
                return {"status": "error", "message": "Failed to get image dimensions"}
        else:
            return {"status": "error", "message": "Either width/height or crop percentages must be specified"}
        
        # Remove virtual canvas info after cropping
        command.extend(["+repage"])
        
        # Add output file
        command.append(output_file)
        
        result = await self._execute_command(command)
        
        return {
            "status": result["status"],
            "input_file": input_file,
            "output_file": output_file,
            "crop_info": crop_info,
            "processing_result": result
        }

    async def _batch_process(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple images."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        input_pattern = config.get("input_pattern")  # e.g., "*.jpg" or ["file1.jpg", "file2.png"]
        input_directory = config.get("input_directory")
        output_directory = data.get("output_directory")
        operation = data.get("operation", "resize")  # resize, convert, filter, crop
        operation_params = data.get("operation_params", {})
        
        if not output_directory:
            raise ValueError("output_directory is required.")
        
        if not input_pattern and not input_directory:
            raise ValueError("Either input_pattern or input_directory must be specified.")
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Get list of files to process
        import glob
        files_to_process = []
        
        if isinstance(input_pattern, list):
            # List of specific files
            files_to_process = input_pattern
        elif input_pattern and input_directory:
            # Pattern in directory
            pattern_path = os.path.join(input_directory, input_pattern)
            files_to_process = glob.glob(pattern_path)
        elif input_pattern:
            # Pattern in current directory
            files_to_process = glob.glob(input_pattern)
        elif input_directory:
            # All image files in directory
            common_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]
            for ext in common_extensions:
                files_to_process.extend(glob.glob(os.path.join(input_directory, ext)))
                files_to_process.extend(glob.glob(os.path.join(input_directory, ext.upper())))
        
        if not files_to_process:
            return {"status": "error", "message": "No files found to process"}
        
        # Process each file
        results = []
        successful_processes = 0
        failed_processes = 0
        
        for input_file in files_to_process:
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_extension = operation_params.get("output_format", "jpg")
                output_file = os.path.join(output_directory, f"{base_name}.{output_extension}")
                
                # Build command based on operation
                if operation == "resize":
                    width = operation_params.get("width", 800)
                    height = operation_params.get("height", 600)
                    maintain_aspect = operation_params.get("maintain_aspect", True)
                    
                    size_param = f"{width}x{height}"
                    if maintain_aspect:
                        size_param += ">"
                    
                    command = [
                        "convert", input_file,
                        "-resize", size_param,
                        output_file
                    ]
                    
                elif operation == "convert":
                    quality = operation_params.get("quality", 90)
                    command = [
                        "convert", input_file,
                        "-quality", str(quality),
                        output_file
                    ]
                    
                elif operation == "filter":
                    filters = operation_params.get("filters", [])
                    command = ["convert", input_file]
                    
                    for filter_config in filters:
                        filter_type = filter_config.get("type")
                        params = filter_config.get("parameters", {})
                        
                        if filter_type == "blur":
                            radius = params.get("radius", 1.0)
                            sigma = params.get("sigma", 1.0)
                            command.extend(["-blur", f"{radius}x{sigma}"])
                        elif filter_type == "sharpen":
                            radius = params.get("radius", 1.0)
                            sigma = params.get("sigma", 1.0)
                            command.extend(["-sharpen", f"{radius}x{sigma}"])
                        elif filter_type == "normalize":
                            command.extend(["-normalize"])
                        elif filter_type == "grayscale":
                            command.extend(["-colorspace", "Gray"])
                    
                    command.append(output_file)
                    
                elif operation == "crop":
                    x = operation_params.get("x", 0)
                    y = operation_params.get("y", 0)
                    width = operation_params.get("width", 100)
                    height = operation_params.get("height", 100)
                    
                    crop_geometry = f"{width}x{height}+{x}+{y}"
                    command = [
                        "convert", input_file,
                        "-crop", crop_geometry,
                        "+repage",
                        output_file
                    ]
                    
                else:
                    results.append({
                        "input_file": input_file,
                        "status": "error",
                        "message": f"Unsupported operation: {operation}"
                    })
                    failed_processes += 1
                    continue
                
                # Execute command
                result = await self._execute_command(command)
                
                if result["status"] == "success":
                    successful_processes += 1
                    results.append({
                        "input_file": input_file,
                        "output_file": output_file,
                        "status": "success",
                        "operation": operation
                    })
                else:
                    failed_processes += 1
                    results.append({
                        "input_file": input_file,
                        "status": "error",
                        "message": result.get("stderr", "Unknown error")
                    })
                    
            except Exception as e:
                failed_processes += 1
                results.append({
                    "input_file": input_file,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "status": "success",
            "operation": operation,
            "total_files": len(files_to_process),
            "successful_processes": successful_processes,
            "failed_processes": failed_processes,
            "output_directory": output_directory,
            "results": results
        }

    # OPENSHOT ACTIONS
    async def _create_project(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new video project."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        project_name = data.get("project_name", "NewProject")
        project_path = data.get("project_path")
        
        # Project settings
        width = data.get("width", 1920)
        height = data.get("height", 1080)
        fps = data.get("fps", 30)
        sample_rate = data.get("sample_rate", 44100)
        channels = data.get("channels", 2)
        
        if not project_path:
            workspace = multimedia_config["workspace_path"]
            project_path = os.path.join(workspace, f"{project_name}.osp")
        
        # Ensure project directory exists
        os.makedirs(os.path.dirname(project_path), exist_ok=True)
        
        # Create OpenShot project file (JSON format)
        project_data = {
            "version": {
                "openshot-qt": "2.6.1",
                "libopenshot": "0.2.7"
            },
            "project": {
                "name": project_name,
                "width": width,
                "height": height,
                "fps": {
                    "num": fps,
                    "den": 1
                },
                "sample_rate": sample_rate,
                "channels": channels,
                "channel_layout": 3 if channels == 2 else 4,
                "settings": {
                    "width": width,
                    "height": height,
                    "fps": {
                        "num": fps,
                        "den": 1
                    },
                    "sample_rate": sample_rate,
                    "channels": channels
                }
            },
            "clips": [],
            "effects": [],
            "files": [],
            "timeline": {
                "tracks": [
                    {
                        "id": 1,
                        "number": 1,
                        "y": 0,
                        "label": "Track 1",
                        "lock": False
                    },
                    {
                        "id": 2,
                        "number": 2,
                        "y": 1,
                        "label": "Track 2",
                        "lock": False
                    }
                ]
            }
        }
        
        # Write project file
        try:
            with open(project_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            return {
                "status": "success",
                "project_name": project_name,
                "project_path": project_path,
                "settings": {
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "sample_rate": sample_rate,
                    "channels": channels
                },
                "message": f"OpenShot project '{project_name}' created successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create project file: {str(e)}"
            }

    async def _add_media(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Add media files to project."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        project_path = config.get("project_path")
        media_files = data.get("media_files", [])
        track_id = data.get("track_id", 1)
        start_time = data.get("start_time", 0.0)
        
        if not project_path:
            raise ValueError("project_path is required.")
        
        if not media_files:
            raise ValueError("media_files list is required.")
        
        if not os.path.exists(project_path):
            return {"status": "error", "message": "Project file not found"}
        
        try:
            # Load existing project
            with open(project_path, 'r') as f:
                project_data = json.load(f)
            
            # Add media files to project
            added_files = []
            added_clips = []
            current_time = start_time
            
            for media_file in media_files:
                file_path = media_file.get("path")
                duration = media_file.get("duration", 5.0)  # default 5 seconds
                
                if not file_path or not os.path.exists(file_path):
                    continue
                
                # Get media info using ffprobe
                probe_cmd = [
                    multimedia_config["ffmpeg_path"].replace("ffmpeg", "ffprobe"),
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    file_path
                ]
                
                probe_result = await self._execute_command(probe_cmd)
                
                if probe_result["status"] == "success":
                    try:
                        probe_data = json.loads(probe_result["stdout"])
                        format_info = probe_data.get("format", {})
                        streams = probe_data.get("streams", [])
                        
                        # Get actual duration if available
                        if "duration" in format_info:
                            actual_duration = float(format_info["duration"])
                        else:
                            actual_duration = duration
                            
                        # Create file entry
                        file_id = len(project_data["files"]) + 1
                        file_entry = {
                            "id": file_id,
                            "path": file_path,
                            "name": os.path.basename(file_path),
                            "duration": actual_duration,
                            "media_type": "video" if any(s.get("codec_type") == "video" for s in streams) else "audio"
                        }
                        
                        # Add video properties if it's a video file
                        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
                        if video_stream:
                            file_entry.update({
                                "width": video_stream.get("width", 1920),
                                "height": video_stream.get("height", 1080),
                                "fps": eval(video_stream.get("r_frame_rate", "30/1")) if video_stream.get("r_frame_rate") else 30
                            })
                        
                        # Add audio properties if it has audio
                        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
                        if audio_stream:
                            file_entry.update({
                                "sample_rate": int(audio_stream.get("sample_rate", 44100)),
                                "channels": int(audio_stream.get("channels", 2))
                            })
                        
                        project_data["files"].append(file_entry)
                        added_files.append(file_entry)
                        
                        # Create clip entry
                        clip_id = len(project_data["clips"]) + 1
                        clip_entry = {
                            "id": clip_id,
                            "file_id": file_id,
                            "track": track_id,
                            "position": current_time,
                            "start": 0,
                            "end": actual_duration,
                            "layer": 0
                        }
                        
                        project_data["clips"].append(clip_entry)
                        added_clips.append(clip_entry)
                        
                        # Move to next position
                        current_time += actual_duration
                        
                    except Exception as e:
                        logger.error(f"Failed to process media file {file_path}: {e}")
                        continue
            
            # Save updated project
            with open(project_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            return {
                "status": "success",
                "project_path": project_path,
                "added_files": len(added_files),
                "added_clips": len(added_clips),
                "track_id": track_id,
                "total_duration": current_time,
                "files": added_files,
                "clips": added_clips
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add media to project: {str(e)}"
            }

    async def _apply_transitions(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply video transitions."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        project_path = config.get("project_path")
        transitions = data.get("transitions", [])
        
        if not project_path:
            raise ValueError("project_path is required.")
        
        if not transitions:
            raise ValueError("transitions list is required.")
        
        if not os.path.exists(project_path):
            return {"status": "error", "message": "Project file not found"}
        
        try:
            # Load existing project
            with open(project_path, 'r') as f:
                project_data = json.load(f)
            
            # Add transitions to project
            added_transitions = []
            
            for transition in transitions:
                transition_type = transition.get("type", "fade")
                duration = transition.get("duration", 1.0)
                position = transition.get("position", 0.0)
                track_id = transition.get("track_id", 1)
                
                # Create transition entry
                transition_id = len(project_data.get("effects", [])) + 1
                transition_entry = {
                    "id": transition_id,
                    "type": "transition",
                    "name": transition_type,
                    "track": track_id,
                    "position": position,
                    "duration": duration,
                    "properties": {
                        "type": transition_type,
                        "brightness": transition.get("brightness", 1.0),
                        "contrast": transition.get("contrast", 1.0)
                    }
                }
                
                if "effects" not in project_data:
                    project_data["effects"] = []
                
                project_data["effects"].append(transition_entry)
                added_transitions.append(transition_entry)
            
            # Save updated project
            with open(project_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            return {
                "status": "success",
                "project_path": project_path,
                "added_transitions": len(added_transitions),
                "transitions": added_transitions
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add transitions to project: {str(e)}"
            }

    async def _add_effects(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Add video effects."""
        multimedia_config = await self._get_multimedia_config(credential_id)
        project_path = config.get("project_path")
        effects = data.get("effects", [])
        
        if not project_path:
            raise ValueError("project_path is required.")
        
        if not effects:
            raise ValueError("effects list is required.")
        
        if not os.path.exists(project_path):
            return {"status": "error", "message": "Project file not found"}
        
        try:
            # Load existing project
            with open(project_path, 'r') as f:
                project_data = json.load(f)
            
            # Add effects to project
            added_effects = []
            
            for effect in effects:
                effect_type = effect.get("type", "brightness")
                clip_id = effect.get("clip_id")
                track_id = effect.get("track_id", 1)
                position = effect.get("position", 0.0)
                duration = effect.get("duration", 5.0)
                
                # Create effect entry
                effect_id = len(project_data.get("effects", [])) + 1
                effect_entry = {
                    "id": effect_id,
                    "type": "effect",
                    "name": effect_type,
                    "clip_id": clip_id,
                    "track": track_id,
                    "position": position,
                    "duration": duration,
                    "properties": {}
                }
                
                # Add type-specific properties
                if effect_type == "brightness":
                    effect_entry["properties"]["brightness"] = effect.get("brightness", 1.0)
                elif effect_type == "contrast":
                    effect_entry["properties"]["contrast"] = effect.get("contrast", 1.0)
                elif effect_type == "saturation":
                    effect_entry["properties"]["saturation"] = effect.get("saturation", 1.0)
                elif effect_type == "blur":
                    effect_entry["properties"]["blur"] = effect.get("blur", 5.0)
                elif effect_type == "crop":
                    effect_entry["properties"].update({
                        "x": effect.get("x", 0),
                        "y": effect.get("y", 0),
                        "width": effect.get("width", 1920),
                        "height": effect.get("height", 1080)
                    })
                elif effect_type == "color_adjustment":
                    effect_entry["properties"].update({
                        "red": effect.get("red", 1.0),
                        "green": effect.get("green", 1.0),
                        "blue": effect.get("blue", 1.0)
                    })
                elif effect_type == "fade":
                    effect_entry["properties"].update({
                        "fade_in": effect.get("fade_in", 0.0),
                        "fade_out": effect.get("fade_out", 0.0)
                    })
                elif effect_type == "scale":
                    effect_entry["properties"].update({
                        "scale_x": effect.get("scale_x", 1.0),
                        "scale_y": effect.get("scale_y", 1.0)
                    })
                elif effect_type == "rotate":
                    effect_entry["properties"]["rotation"] = effect.get("rotation", 0.0)
                
                if "effects" not in project_data:
                    project_data["effects"] = []
                
                project_data["effects"].append(effect_entry)
                added_effects.append(effect_entry)
            
            # Save updated project
            with open(project_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            return {
                "status": "success",
                "project_path": project_path,
                "added_effects": len(added_effects),
                "effects": added_effects
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add effects to project: {str(e)}"
            }

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
        multimedia_config = await self._get_multimedia_config(credential_id)
        project_path = config.get("project_path")
        timeline_config = data.get("timeline_config", {})
        
        if not project_path:
            raise ValueError("project_path is required.")
        
        # Timeline configuration
        tracks = timeline_config.get("tracks", [])
        duration = timeline_config.get("duration", 60)  # seconds
        fps = timeline_config.get("fps", 30)
        
        if not tracks:
            raise ValueError("At least one track is required in timeline_config.")
        
        try:
            # Create a comprehensive timeline configuration
            timeline_data = {
                "project_path": project_path,
                "duration": duration,
                "fps": fps,
                "tracks": [],
                "total_tracks": len(tracks)
            }
            
            for track_idx, track in enumerate(tracks):
                track_type = track.get("type", "video")  # video, audio, image
                track_name = track.get("name", f"Track_{track_idx + 1}")
                clips = track.get("clips", [])
                
                track_data = {
                    "track_id": track_idx + 1,
                    "track_name": track_name,
                    "track_type": track_type,
                    "clips": []
                }
                
                current_position = 0
                for clip_idx, clip in enumerate(clips):
                    clip_path = clip.get("path")
                    clip_duration = clip.get("duration", 5)  # seconds
                    clip_start = clip.get("start", current_position)
                    clip_end = clip_start + clip_duration
                    
                    # Clip effects and transitions
                    effects = clip.get("effects", [])
                    transitions = clip.get("transitions", [])
                    
                    clip_data = {
                        "clip_id": clip_idx + 1,
                        "clip_path": clip_path,
                        "start_time": clip_start,
                        "end_time": clip_end,
                        "duration": clip_duration,
                        "effects": effects,
                        "transitions": transitions
                    }
                    
                    # Add clip-specific properties based on type
                    if track_type == "video":
                        clip_data.update({
                            "x": clip.get("x", 0),
                            "y": clip.get("y", 0),
                            "width": clip.get("width", 1920),
                            "height": clip.get("height", 1080),
                            "scale": clip.get("scale", 1.0),
                            "rotation": clip.get("rotation", 0)
                        })
                    elif track_type == "audio":
                        clip_data.update({
                            "volume": clip.get("volume", 1.0),
                            "fade_in": clip.get("fade_in", 0),
                            "fade_out": clip.get("fade_out", 0),
                            "channels": clip.get("channels", 2)
                        })
                    elif track_type == "image":
                        clip_data.update({
                            "x": clip.get("x", 0),
                            "y": clip.get("y", 0),
                            "width": clip.get("width", 1920),
                            "height": clip.get("height", 1080),
                            "scale": clip.get("scale", 1.0),
                            "rotation": clip.get("rotation", 0),
                            "opacity": clip.get("opacity", 1.0)
                        })
                    
                    track_data["clips"].append(clip_data)
                    current_position = clip_end
                
                timeline_data["tracks"].append(track_data)
            
            # Generate timeline script for OpenShot
            script = self._generate_openshot_timeline_script(timeline_data)
            
            # Execute the timeline generation
            result = await self._execute_command(
                multimedia_config["workspace_path"],
                ["python3", "-c", script],
                "generate_timeline"
            )
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "timeline_generated": True,
                    "project_path": project_path,
                    "tracks_count": len(tracks),
                    "total_duration": duration,
                    "fps": fps,
                    "timeline_data": timeline_data
                }
            else:
                return {
                    "status": "error",
                    "message": f"Timeline generation failed: {result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"Failed to generate timeline: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to generate timeline: {str(e)}"}
    
    def _generate_openshot_timeline_script(self, timeline_data: Dict[str, Any]) -> str:
        """Generate Python script for OpenShot timeline creation"""
        script = f"""
import openshot
import json
import os

# Create project
project_path = "{timeline_data['project_path']}"
os.makedirs(os.path.dirname(project_path), exist_ok=True)

# Initialize OpenShot project
project = openshot.Project(project_path)

# Set project properties
project.info.fps = {timeline_data['fps']}
project.info.duration = {timeline_data['duration']}
project.info.width = 1920
project.info.height = 1080
project.info.sample_rate = 44100
project.info.channels = 2

# Create timeline
timeline = openshot.Timeline(project.info)

# Add tracks
"""
        
        for track in timeline_data['tracks']:
            script += f"""
# Add {track['track_type']} track: {track['track_name']}
track_{track['track_id']} = openshot.Track()
track_{track['track_id']}.info.number = {track['track_id']}
track_{track['track_id']}.info.label = "{track['track_name']}"
timeline.AddTrack(track_{track['track_id']})

"""
            
            for clip in track['clips']:
                if clip['clip_path']:
                    script += f"""
# Add clip: {clip['clip_path']}
clip_{clip['clip_id']} = openshot.Clip("{clip['clip_path']}")
clip_{clip['clip_id']}.Position({clip['start_time']})
clip_{clip['clip_id']}.Start(0)
clip_{clip['clip_id']}.End({clip['duration']})
clip_{clip['clip_id']}.Layer({track['track_id']})

"""
                    
                    # Add clip-specific properties
                    if track['track_type'] == 'video':
                        script += f"""
# Video clip properties
clip_{clip['clip_id']}.location_x = openshot.Point({clip['x']}, {clip['x']})
clip_{clip['clip_id']}.location_y = openshot.Point({clip['y']}, {clip['y']})
clip_{clip['clip_id']}.scale_x = openshot.Point({clip['scale']}, {clip['scale']})
clip_{clip['clip_id']}.scale_y = openshot.Point({clip['scale']}, {clip['scale']})
clip_{clip['clip_id']}.rotation = openshot.Point({clip['rotation']}, {clip['rotation']})
"""
                    elif track['track_type'] == 'audio':
                        script += f"""
# Audio clip properties
clip_{clip['clip_id']}.volume = openshot.Point({clip['volume']}, {clip['volume']})
"""
                    
                    script += f"""
timeline.AddClip(clip_{clip['clip_id']})

"""
        
                 script += f"""
# Save project
project.Save()

print("Timeline generated successfully")
print(f"Project saved to: {{project_path}}")
print(f"Total tracks: {timeline_data['total_tracks']}")
print(f"Duration: {timeline_data['duration']} seconds")
print(f"FPS: {timeline_data['fps']}")
"""
        
        return script


# Instantiate a single instance
multimedia_connector = MultimediaConnector() 