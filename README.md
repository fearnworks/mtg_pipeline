# Use Case: Magic: The Gathering Card Identification and Archiving from Video

## Summary:
We aim to develop an application that processes prerecorded videos of Magic: The Gathering (MTG) cards, identifies each card displayed in the video frame, extracts relevant textual information, and stores this data in a structured format. This digital archive will facilitate the creation and management of MTG card collections and aid players in building and sharing decks.

## Actors:

- MTG Players/Collectors
- Deck Builders
- Archivists

## Preconditions:

The application is provided with prerecorded videos of MTG cards.
The videos are clear, with the cards placed such that the textual information is visible.
The necessary computational resources and permissions for processing videos and storing data are available.

## Basic Flow:

1. The user provides a prerecorded video file as input to the application.
2. The application processes the video, extracting one frame per second to reduce redundancy.
3. For each extracted frame, the application uses a machine learning model to detect the presence and boundaries of MTG cards.
4. Once a card is detected, the application identifies and extracts the region containing the card's name and other textual information.
5. The application preprocesses the extracted region to enhance text visibility for OCR (Optical Character Recognition).
6. OCR is performed on the preprocessed image to extract the card's textual data, such as its name, edition, and collector number.
7. The extracted information is added to a digital archive in a structured format, which could be a database or a spreadsheet.
8. If a card is displayed for multiple seconds in the video, the application ensures that the card's information is archived only once to prevent duplicates.
9. The user is provided with feedback regarding the number of cards processed and any errors or uncertainties encountered during identification.

## Postconditions:

A digital archive of the MTG cards presented in the video is created.
The archive includes the names and relevant textual data of the cards.
The user can access and utilize the archived data for deck building or collection management.

## Exception Scenarios:

If a card cannot be identified due to poor video quality or occlusion, the application logs this event and may flag it for manual review.
If OCR cannot reliably extract text, the application returns a placeholder value (e.g., "No card") and may request human intervention for accurate data entry.

## Success Guarantee:

The application will accurately identify and extract data from the majority of MTG cards presented in the input video.
The digital archive will be comprehensive and formatted for ease of use in subsequent deck building or archiving activities.

## DataFrame Columns Description

- `object`: The type of object it is (card, error, etc).
- `id`: A unique ID for the returned card object.
- `name`: The oracle name of the card.
- `uri`: The Scryfall API uri for the card.
- `scryfall_uri`: The full Scryfall page of the card.
- `layout`: The image layout of the card. (normal, transform, etc).
- `highres_image`: Determine if a card has a highres scan available.
- `cmc`: A float of the converted mana cost of the card.
- `type_line`: The full type line of the card.
- `oracle_text`: The official oracle text of a card.
- `mana_cost`: The full mana cost using shorthanded mana symbols.
- `colors`: A list of strings with all colors found in the mana cost.
- `color_identity`: A list of strings with all colors found on the card itself.
- `legalities`: A dictionary of all formats and their legality.
- `reserved`: Returns True if the card is on the reserved list.
- `reprint`: Returns True if the card has been reprinted before.
- `set_code`: The 3 letter code for the set of the card.
- `set_name`: The full name for the set of the card.
- `set_uri`: The API uri for the full set list of the card.
- `set_search_uri`: Same output as set_uri.
- `scryfall_set_uri`: The full link to the set on Scryfall.
- `rulings_uri`: The API uri for the rulings of the card.
- `prints_search_uri`: A link to where you can begin paginating all re/prints for this card on Scryfall’s API.
- `collector_number`: The collector number of the card.
- `digital`: Returns True if the card is the digital version.
- `rarity`: The rarity of the card.
- `illustration_id`: The related id of the card art.
- `artist`: The artist of the card.
- `frame`: The year of the card frame.
- `frame_effects`: The card's frame effect, if any. (miracle, nyxtouched, etc.)
- `full_art`: Returns True if the card is considered full art.
- `border_color`: The color of the card border.
- `edhrec_rank`: The rank of the card on edhrec.com.
- `prices`: The prices of the card in USD.
- `related_uris`: The related URIs of the card.
- `purchase_uris`: The purchase URIs of the card.
- `story_spotlight`: The story spotlight of the card.
- `power`: The power of the card.
- `toughness`: The toughness of the card.
- `arena_id`: The arena ID of the card.
- `lang`: The language of the card.
- `oracle_id`: The oracle ID of the card.
- `oversized`: Returns True if the card is oversized.
- `games`: The games the card is part of.
- `promo`: Returns True if the card is a promo.
- `released_at`: The release date of the card.
- `preview`: The preview of the card.
- `image_status`: The image status of the card.
- `finishes`: The finishes of the card.

# MTG Card Detector Training Utility Guide

The `training_util.py` script facilitates various stages of the ML lifecycle for Magic: The Gathering (MTG) card detection. Below is a guide on how to use this utility.

## Commands

### Training a Model

To train a model, you need to specify the model configuration file, the data configuration file, and the number of epochs.

```bash
python training_util.py train --config yolov8n.yaml --data ./training/config.yaml --epochs 10
```

### Predicting Using a Trained Model

To predict using a trained model, you need to provide the path to the video file, the trained model file, and the output directory to save the processed video.

```bash
python training_util.py predict -v /path/to/video.mp4 --model /path/to/model.pt -o /path/to/output
```

## Extracting Frames from a Video
To extract frames from a video for dataset creation, you need the video path and the output directory. You can also specify the number of seconds to skip between frames and whether to rotate the video.


```bash
python training_util.py extract_frames --video_path /path/to/video.mp4 --output_dir /path/to/frames --skip_rate 1 --rotate
```