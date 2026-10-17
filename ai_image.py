import os
import sys
import argparse
from io import BytesIO
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types


# Default prompt based on your specification, refined so skeletonize_mask()
# in image_to_svg.py gets clean, single-stroke black lines on pure white.
DEFAULT_PRIMAL_PROMPT = (
    "Create a drawing of this image using simple lines and curves. "
    "Keep important details such as text"
    "No shading, no cross-hatching, and no solid black fills. "
    "Keep the shapes, proportions, and positions of all the objects in the image the same. "
    "Make it a minimalist primal sketch using clean, uniform-thickness black ink lines "
    "on a pure solid white (#FFFFFF) background."
    "make the lines stand out as much as possible being solid black, no transparency"
)

# Even stricter prompt if you want ultra-simple outlines with minimal internal detail
MINIMAL_PRIMAL_PROMPT = (
    "Create an ultra-minimalist primal sketch of this image using as few simple continuous "
    "black lines and smooth curves as possible. Capture only the primary outer silhouettes "
    "and most essential structural features. No text, no background clutter, no shading, "
    "and no solid fills. Pure black lines on a pure solid white (#FFFFFF) background."
)


def generate_nanobanana_sketch(
    input_image_path: str,
    output_image_path: str,
    prompt: str = DEFAULT_PRIMAL_PROMPT,
    model_id: str = "gemini-2.5-flash-image",
    api_key: str = None,
    match_original_size: bool = True
) -> str:
    """Sends an input image to Nano Banana (Gemini Flash Image) to generate
    a clean black-on-white primal sketch PNG ready for image_to_svg.py.

    Parameters:
        input_image_path (str): Path to the source image (e.g., images/orb.png).
        output_image_path (str): Path where the primal sketch PNG will be saved.
        prompt (str): Instruction telling Nano Banana how to draw the sketch.
        model_id (str): 'gemini-2.5-flash-image' (Nano Banana) or
                        'gemini-3.1-flash-image' (Nano Banana 2).
        api_key (str): Optional Gemini API key (defaults to GEMINI_API_KEY env var).
        match_original_size (bool): Resize output back to exact input (W, H).

    Returns:
        str: Path to the saved primal sketch PNG.
    """
    input_path = Path(input_image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input image: {input_image_path}")

    # 1. Load the original image and record its exact pixel dimensions
    source_img = Image.open(input_path)
    orig_width, orig_height = source_img.size
    print(f"[1/3] Loaded '{input_image_path}' ({orig_width}x{orig_height} px).")

    # 2. Initialize the Gemini client
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "No API key found! Set GEMINI_API_KEY in your environment:\n"
            "  export GEMINI_API_KEY='your_key_here'\n"
            "or pass it directly with --api-key."
        )

    client = genai.Client(api_key=resolved_key)

    print(f"[2/3] Asking Nano Banana ({model_id}) to generate the primal sketch...")
    response = client.models.generate_content(
        model=model_id,
        contents=[source_img, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # 3. Extract the generated image from the response parts
    sketch_img = None
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                sketch_img = Image.open(BytesIO(part.inline_data.data))
                break

    if sketch_img is None:
        # Print text explanation if the model returned text instead of an image
        text_Output = [p.text for p in response.candidates[0].content.parts if p.text]
        raise RuntimeError(
            f"Nano Banana did not return an image. Model response: {' '.join(text_Output)}"
        )

    # 4. Resize back to match the original image dimensions so (x, y) coordinates
    # in image_to_svg.py match the original image for your Step 1 metric
    if match_original_size and sketch_img.size != (orig_width, orig_height):
        print(f"      Resizing sketch from {sketch_img.size} -> {(orig_width, orig_height)} to match original canvas...")
        sketch_img = sketch_img.resize((orig_width, orig_height), Image.Resampling.LANCZOS)

    # Ensure output directory exists and save as PNG
    out_path = Path(output_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sketch_img.save(out_path, format="PNG")
    print(f"[3/3] Saved primal sketch to: {out_path}")

    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use Nano Banana (Gemini Image API) to turn any image into a clean Primal Sketch PNG."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="images/orb.png",
        help="Path to input image (default: images/orb.png)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG path (default: <input_name>_sketch.png)"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use the stricter minimal prompt (strips text and internal details)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default=None,
        help="Custom prompt override for Nano Banana"
    )
    parser.add_argument(
        "-m", "--model",
        default="gemini-2.5-flash-image",
        help="Model ID (default: 'gemini-2.5-flash-image', or try 'gemini-3.1-flash-image')"
    )
    parser.add_argument(
        "-k", "--api-key",
        default=None,
        help="Gemini API Key (optional if GEMINI_API_KEY env var is set)"
    )

    args = parser.parse_args()

    # Automatically set output filename to <input_name>_sketch.png if not specified
    input_path = Path(args.input)
    if args.output is None:
        args.output = str(input_path.with_name(f"{input_path.stem}_sketch.png"))

    # Select prompt
    chosen_prompt = args.prompt
    if chosen_prompt is None:
        chosen_prompt = MINIMAL_PRIMAL_PROMPT if args.minimal else DEFAULT_PRIMAL_PROMPT

    try:
        saved_sketch = generate_nanobanana_sketch(
            input_image_path=args.input,
            output_image_path=args.output,
            prompt=chosen_prompt,
            model_id=args.model,
            api_key=args.api_key
        )
        print(f"\nNext step -> Run your SVG converter on this file:")
        print(f"  python image_to_svg.py {saved_sketch} -o {input_path.with_name(f'{input_path.stem}_output.svg')}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)