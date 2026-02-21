# OCR

Extract text from images and PDFs using HuggingFace image-to-text models.

## Parameters

| Parameter      | Default                        | Description                                    |
|----------------|--------------------------------|------------------------------------------------|
| `--model`      | `microsoft/trocr-base-printed` | HuggingFace model repo ID                      |
| `--revision`   | `main`                         | Model revision (branch, tag, or commit hash)   |
| `--cache-dir`  |                                | HuggingFace cache directory for model files    |
| `--device`     | `auto`                         | Device to use (`cuda`, `cpu`, or `auto`)       |
| `--max-length` | `512`                          | Maximum number of tokens to generate per image |
| `--batch-size` | `4`                            | Number of images to process in parallel on GPU |

## Supported Input Formats

- Image files (PNG, JPEG, TIFF, etc.)
- PDF files (each page is rendered and processed separately)

## Output Format

Plain text. For multi-page inputs (PDFs), pages are separated by form-feed characters (`\f`).

## Models

Any HuggingFace [`image-to-text`](https://huggingface.co/models?pipeline_tag=image-to-text) model is supported.

### Printed text

| Model | Params | Description | License |
|-------|--------|-------------|---------|
| [`microsoft/trocr-base-printed`](https://huggingface.co/microsoft/trocr-base-printed) (default) | 334M | General-purpose printed text recognition | MIT |
| [`microsoft/trocr-large-printed`](https://huggingface.co/microsoft/trocr-large-printed) | 558M | Higher accuracy for printed text | MIT |

### Handwritten text

| Model | Params | Description | License |
|-------|--------|-------------|---------|
| [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) | 334M | General-purpose handwriting recognition | MIT |
| [`microsoft/trocr-large-handwritten`](https://huggingface.co/microsoft/trocr-large-handwritten) | 558M | Higher accuracy for handwritten text | MIT |

## Examples

### Extract text from scanned images

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: ocr
        kind: local
        module: tigerflow_ml.text.ocr.local
        input_ext: .png
        output_ext: .txt
    ```

=== "Input"

    A scanned image of a printed document, e.g. `invoice.png`.

=== "Output"

    ```text title="invoice.txt"
    INVOICE
    Date: 2025-01-15
    Invoice Number: INV-2025-0042

    Bill To:
    Jane Doe
    123 Main Street
    Princeton, NJ 08544
    ```

### Extract text from a multi-page PDF

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: ocr
        kind: local
        module: tigerflow_ml.text.ocr.local
        input_ext: .pdf
        output_ext: .txt
        params:
          batch_size: 8
    ```

=== "Input"

    A scanned PDF document, e.g. `report.pdf` with 3 pages.

=== "Output"

    ```text title="report.txt"
    Page one text content here...
    ␌
    Page two text content here...
    ␌
    Page three text content here...
    ```

    Each page is separated by a form-feed character (`\f`, shown as `␌`).

!!! tip

    For large PDFs with many pages, increase `--batch-size` to process more pages
    in parallel on the GPU.
