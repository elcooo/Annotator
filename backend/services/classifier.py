import sqlalchemy.orm as orm
import models
import schemas
import fastapi
import datetime as dt
import os
import pandas as pd
import random

# from preprocessing.classifier import pipeline, predict
from preprocessing.classifier import pipeline, predict

from services.segments import segment_selector
from services.labels import get_labels
import services.prototypes as prototype_services

async def audio_selector(filename: str, user: schemas.User, db: orm.Session):
    audio = db.query(models.Audio).filter_by(owner_id=user.id).filter(models.Audio.filename == filename).first()
    if audio is None:
        raise fastapi.HTTPException(status_code=404, detail="Audio doesn't exist")
    return audio

# Change Status
async def change_status(start: str, end: str, status: str, user: schemas.User, db: orm.Session):
    audio_files = db.query(models.Audio).filter_by(owner_id=user.id).filter(models.Audio.status == 'Complete').filter(models.Audio.validation != True).all()
    startDate = dt.datetime(*list(map(int, start.split('-'))))
    endDate = dt.datetime(*list(map(int, end.split('-'))))

    for audio in audio_files:
        dateCreated = dt.datetime(*list(map(int, audio.date_created.split('-'))))
        if ((dateCreated >= startDate) and (dateCreated <= endDate)):
            instance_db = db.query(models.Audio).filter_by(owner_id=user.id).filter(models.Audio.filename == audio.filename).first()
            instance_db.status = status
            db.commit()
            db.refresh(instance_db)
    return "Status Updated"


async def export(user: schemas.User, db: orm.Session, start:str='2020-01-01', end:str='2200-01-01'):
    print(f"[EXPORT] Starting export process for user {user.id}")
    print(f"[EXPORT] Date range: {start} to {end}")
    
    await create_validation(user, db)
    segments = db.query(models.Segments).filter_by(owner_id=user.id).all() # .filter(models.Segments.status == 'Complete')
    print(f"[EXPORT] Found {len(segments)} total segments in database")
    
    startDate = dt.datetime(*list(map(int, start.split('-'))))
    endDate = dt.datetime(*list(map(int, end.split('-'))))

    training_set = []
    columns = ['Filename', 'Validation', 'Label', 'Start', 'End']

    MODEL_PATH = f'./static/{user.id}/model'
    if not os.path.exists(MODEL_PATH):
        print(f"[EXPORT] Creating model directory: {MODEL_PATH}")
        os.makedirs(MODEL_PATH, exist_ok=True)
    else:
        print(f"[EXPORT] Model directory already exists: {MODEL_PATH}")

    print(f"[EXPORT] Filtering segments by date range...")
    for segment in segments:
        dateCreated = dt.datetime(*list(map(int, segment.date_created.split('-'))))

        if ((dateCreated >= startDate) and (dateCreated <= endDate)):
            seg = [segment.filename, 
                   segment.validation, 
                   segment.label, 
                   segment.start,
                   segment.end]
            training_set.append(seg)

    print(f"[EXPORT] Filtered to {len(training_set)} segments in date range")
    df_training = pd.DataFrame(training_set, columns=columns)
    
    annotations_path = f"./static/{user.id}/model/annotations.csv"
    print(f"[EXPORT] Saving annotations to: {annotations_path}")
    df_training.to_csv(annotations_path)
    print(f"[EXPORT] Annotations saved successfully")
    
    result = df_training.to_dict()
    print(f"[EXPORT] Export completed, returning {len(result['Filename'])} training samples")
    return result

async def create_validation(user: schemas.User, db: orm.Session):
    print(f"[VALIDATION] Starting validation dataset creation for user {user.id}")
    # Sets validation dataset, maintains 20/80 val-training split
    segments = db.query(models.Segments).filter_by(owner_id=user.id).filter(models.Segments.status == 'Complete').all()
    validation = db.query(models.Segments).filter_by(owner_id=user.id).filter(models.Segments.status == 'Complete').filter(models.Segments.validation == True).all()
    
    print(f"[VALIDATION] Found {len(segments)} complete segments")
    print(f"[VALIDATION] Found {len(validation)} existing validation segments")

    def prob(segments, validation, target=0.2):
        p = target - len(validation)/len(segments)
        return p if (p > 0) else None

    def decision(probability):
        return random.random() < probability
    
    if len(segments) > 0:
        p = prob(segments, validation)
        print(f"[VALIDATION] Target validation ratio: 20%, current ratio: {len(validation)/len(segments)*100:.1f}%")
        print(f"[VALIDATION] Probability for new validation samples: {p}")
    else:
        p = None
        print(f"[VALIDATION] No segments found, skipping validation assignment")

    if p:
        new_validation_count = 0
        for segment in segments:
            if decision(p): 
                segment_db = await segment_selector(segment.filename, user, db)
                if segment_db:
                    segment_db.validation = True
                    db.commit()
                    db.refresh(segment_db)
                    new_validation_count += 1
                    print(f"[VALIDATION] Marked segment {segment.filename} as validation")
                else:
                    print(f"[VALIDATION] Warning: Could not find segment {segment.filename} in database")
        print(f"[VALIDATION] Added {new_validation_count} new validation samples")
    else:
        print(f"[VALIDATION] No new validation samples needed")
    
    print(f"[VALIDATION] Validation dataset creation completed")
    return segments

async def classifier(user: schemas.User, db: orm.Session, num_epoch=25, batch_size=12):
    print(f"[CLASSIFIER] Starting classifier process for user {user.id}")
    
    print(f"[CLASSIFIER] Creating validation dataset...")
    await create_validation(user, db)
    
    print(f"[CLASSIFIER] Exporting training data...")
    segs = await export(user, db)

    print(f"[CLASSIFIER] Found {len(segs['Filename'])} training samples")
    if len(segs['Filename']) < 10:
        print(f"[CLASSIFIER] ERROR: Insufficient training data ({len(segs['Filename'])} < 10)")
        raise fastapi.HTTPException(status_code=405, detail="Insufficient training data")
    else:
        print(f"[CLASSIFIER] Training data sufficient, proceeding with training setup")
        MODEL_PATH = f'./static/{user.id}/model'
        if not os.path.exists(MODEL_PATH):
            print(f"[CLASSIFIER] Creating model directory: {MODEL_PATH}")
            os.makedirs(MODEL_PATH, exist_ok=True)
        else:
            print(f"[CLASSIFIER] Model directory already exists: {MODEL_PATH}")

        print(f"[CLASSIFIER] Getting user labels...")
        classes = await get_labels(user, db)
        labels = []
        for label in classes.keys():
            labels.append(label)
        print(f"[CLASSIFIER] Found {len(labels)} classes: {labels}")

        # Run training synchronously to get real-time results
        print(f"[CLASSIFIER] Starting training pipeline...")
        try:
            loss, acc, val_loss, val_acc = pipeline(user, labels, num_epoch, batch_size)
            print(f"[CLASSIFIER] Training completed successfully")
            
            # Set all segments to trained
            print(f"[CLASSIFIER] Updating segment statuses to 'Trained'...")
            segments = db.query(models.Segments).filter_by(owner_id=user.id).filter(models.Segments.status == 'Complete').all()
            print(f"[CLASSIFIER] Found {len(segments)} segments to mark as trained")
            for segment in segments:
                segment_db = await segment_selector(segment.filename, user, db)
                segment_db.status = "Trained"
                db.commit()
                db.refresh(segment_db)
            print(f"[CLASSIFIER] All segments marked as trained")

            # Update prototypes
            print(f"[CLASSIFIER] Generating prototypes...")
            await prototype_services.generate_supports(user, db)
            print(f"[CLASSIFIER] Prototypes generated successfully")

            # Return real training statistics
            stats = {
                'loss': loss,
                'accuracy': acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            print(f"[CLASSIFIER] Returning real training stats: {stats}")
            return stats
            
        except Exception as e:
            print(f"[CLASSIFIER] Training failed with error: {e}")
            raise fastapi.HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

async def prediction(segment, user: schemas.User, db: orm.Session):
    classes = await get_labels(user, db)
    labels = []
    for label in classes.keys():
        labels.append(label)

    pred, confidence = predict(segment, user, labels)

    segment_db = await segment_selector(segment, user, db)

    if confidence < 0.7:
        segment_db.label = 'unknown'
    else:
        segment_db.label = pred 
        
    segment_db.status = "Automatic"
    segment_db.confidence = confidence
    db.commit()
    db.refresh(segment_db)

    return pred

