import asyncio
from data.parsers import apple_watch_parser, bioradio_parser, csv_parser
from pipeline.timing import timed

async def run_session(session_dir: str) -> dict:
    # Parse all 3 sources concurrently — no waiting on one before the other
    with timed("ingest"):
        xml_data, bcrx_data, csv_data = await asyncio.gather(
            asyncio.to_thread(apple_watch_parser.load, session_dir),
            asyncio.to_thread(bioradio_parser.load, session_dir),
            asyncio.to_thread(csv_parser.load, session_dir),
        )

    with timed("validate + sync"):
        aligned = validate_and_sync(xml_data, bcrx_data, csv_data)

    with timed("feature extraction"):
        features = build_feature_stack(aligned)

    with timed("model inference"):
        score = regressor.predict(features)        # embodiment 0–100
        classes = classifier.predict(features)     # prosthetic classes

    return {"embodiment_score": score, "prosthetic_classes": classes}