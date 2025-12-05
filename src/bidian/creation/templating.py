"""Handles loading and rendering note templates."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
from jinja2 import Environment, Template, TemplateError, UndefinedError
from yaml import YAMLError

logger = logging.getLogger(__name__)

# Define default and custom template directories
# Use Path(__file__).parent to get the directory of the current file
MODULE_DIR = Path(__file__).parent
DEFAULT_TEMPLATES_DIR = MODULE_DIR / "default_templates"
# Allow user override via environment variable or config later
CUSTOM_TEMPLATES_DIR_PATH = os.environ.get(
    "BIDIAN_CUSTOM_TEMPLATES_PATH", "./templates")
CUSTOM_TEMPLATES_DIR = Path(CUSTOM_TEMPLATES_DIR_PATH)


class TemplateRenderingError(Exception):
    """Custom exception for errors during template rendering."""
    pass


class TemplateNotFoundError(Exception):
    """Custom exception when a template cannot be found."""
    pass


class TemplateRenderer:
    """Loads YAML/Jinja2 templates and renders them to create note content."""

    def __init__(self,
                 custom_template_dir: Path = CUSTOM_TEMPLATES_DIR,
                 default_template_dir: Path = DEFAULT_TEMPLATES_DIR):
        """Initializes the renderer.

        Args:
            custom_template_dir: Path to the directory with custom user templates.
            default_template_dir: Path to the directory with default templates.
        """
        self.custom_dir = custom_template_dir
        self.default_dir = default_template_dir
        # Basic Jinja environment - can be customized further if needed
        # No HTML autoescaping for Markdown
        self.jinja_env = Environment(autoescape=False)
        logger.info(
            f"TemplateRenderer initialized. Custom dir: '{self.custom_dir}', Default dir: '{self.default_dir}'")

    def _find_template_path(self, template_name: str) -> Optional[Path]:
        """Finds the path to a template, prioritizing the custom directory."""
        if not template_name.endswith('.yaml') and not template_name.endswith('.yml'):
            template_name += ".yaml"  # Assume .yaml if no extension

        custom_path = self.custom_dir / template_name
        if custom_path.is_file():
            logger.debug(f"Using custom template: {custom_path}")
            return custom_path

        default_path = self.default_dir / template_name
        if default_path.is_file():
            logger.debug(f"Using default template: {default_path}")
            return default_path

        logger.error(
            f"Template '{template_name}' not found in {self.custom_dir} or {self.default_dir}")
        return None

    def render_template(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Loads a template, renders it with context, and returns the full note string.

        Args:
            template_name: The name of the template file (e.g., "hub_page").
            context: A dictionary containing variables for the template 
                     (e.g., {'title': '...', 'outline': [...], 'backlinks': [...]}).

        Returns:
            The fully rendered note content as a string, including front-matter.

        Raises:
            TemplateNotFoundError: If the template file cannot be found.
            TemplateRenderingError: If there's an error loading or rendering the template.
        """
        template_path = self._find_template_path(template_name)
        if not template_path:
            raise TemplateNotFoundError(f"Template '{template_name}' not found.")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
        except (OSError, YAMLError) as e:
            logger.error(
                f"Error loading template file {template_path}: {e}", exc_info=True)
            raise TemplateRenderingError(
                f"Failed to load template '{template_name}': {e}") from e

        if not isinstance(template_data, dict):
            raise TemplateRenderingError(
                f"Template file '{template_path}' does not contain a valid YAML dictionary.")

        # Extract front_matter and content template
        default_front_matter = template_data.get('front_matter', {})
        content_template_str = template_data.get('content', '')

        # Render the main content using Jinja2
        try:
            template: Template = self.jinja_env.from_string(content_template_str)
            rendered_content = template.render(context)
        except (TemplateError, UndefinedError) as e:
            logger.error(
                f"Error rendering content template '{template_name}': {e}", exc_info=True)
            raise TemplateRenderingError(
                f"Failed to render content for '{template_name}': {e}") from e

        # --- Construct final output ---
        final_note_parts = []

        # 1. Front Matter
        # Merge default front matter with any front matter provided in the context,
        # giving context precedence.
        final_front_matter = default_front_matter.copy()
        context_fm = context.get('front_matter', {})
        if isinstance(context_fm, dict):
            final_front_matter.update(context_fm)
        else:
            logger.warning(
                f"Context 'front_matter' is not a dict, ignoring: {context_fm}")

        # Only add front matter block if there is any
        if final_front_matter:
            try:
                # Use safe_dump, ensure proper YAML formatting
                fm_yaml = yaml.safe_dump(
                    final_front_matter, default_flow_style=False, sort_keys=False)
                final_note_parts.append("---")
                final_note_parts.append(fm_yaml.strip())
                final_note_parts.append("---")
            except YAMLError as e:
                logger.error(
                    f"Error formatting final front matter for '{template_name}': {e}", exc_info=True)
                # Proceed without front matter? Or raise?
                raise TemplateRenderingError(
                    f"Failed to format front matter for '{template_name}': {e}") from e

        # 2. Rendered Content
        final_note_parts.append(rendered_content.strip())

        # Join parts, ensuring separation between FM and content
        full_note = "\n".join(final_note_parts)
        # Ensure a single trailing newline for consistency
        return full_note.strip() + '\n'

# Example Usage:
# if __name__ == '__main__':
#     # Setup basic logging
#     logging.basicConfig(level=logging.INFO)
#
#     # Create dummy default template
#     DEFAULT_TEMPLATES_DIR.mkdir(exist_ok=True)
#     hub_template_content = """
# front_matter:
#   type: hub
#   status: wip
#   tags: ["map-of-content"]
# content: |
#   # {{ title | default('Hub Page') }}
#
#   This page connects related concepts.
#
#   ## Outline
#   {% if outline %}
#   {% for item in outline %}
#   - {{ item }}
#   {% endfor %}
#   {% else %}
#   (Outline TBD)
#   {% endif %}
#
#   ## Related Notes
#   {% if backlinks %}
#   {% for link in backlinks %}
#   - [[{{ link }}]]
#   {% endfor %}
#   {% else %}
#   (No specific backlinks identified yet)
#   {% endif %}
#
#   {{ additional_content | default('') }}
# """
#     (DEFAULT_TEMPLATES_DIR / "hub_page.yaml").write_text(hub_template_content)
#
#     # Create dummy custom template directory
#     CUSTOM_TEMPLATES_DIR.mkdir(exist_ok=True)
#     custom_template_content = """
# front_matter:
#   custom: true
# content: |
#   # Custom {{ title }}
#   Rendered by custom template.
# """
#     (CUSTOM_TEMPLATES_DIR / "custom_test.yaml").write_text(custom_template_content)
#
#     renderer = TemplateRenderer()
#
#     # Test rendering default template
#     print("--- Rendering Default Hub Page ---")
#     hub_context = {
#         'title': 'AI Concepts Hub',
#         'outline': ['Machine Learning', 'Deep Learning', 'NLP'],
#         'backlinks': ['Notes/ML Intro', 'Notes/Transformer Arch'],
#         'front_matter': {'status': 'draft'} # Override default status
#     }
#     try:
#         rendered_hub = renderer.render_template('hub_page', hub_context)
#         print(rendered_hub)
#     except Exception as e:
#         print(f"Error: {e}")
#
#     # Test rendering custom template
#     print("\n--- Rendering Custom Template ---")
#     custom_context = {'title': 'My Custom Note'}
#     try:
#         rendered_custom = renderer.render_template('custom_test', custom_context)
#         print(rendered_custom)
#     except Exception as e:
#         print(f"Error: {e}")
#
#     # Test template not found
#     print("\n--- Rendering Non-existent Template ---")
#     try:
#         renderer.render_template('nonexistent', {})
#     except TemplateNotFoundError as e:
#         print(f"Caught expected error: {e}")
#     except Exception as e:
#         print(f"Caught unexpected error: {e}")
