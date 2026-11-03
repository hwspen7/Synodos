"""Size-limited media ingestion and source-aware extraction."""
import hashlib
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

SUPPORTED = {'.txt','.md','.pdf','.docx','.pptx','.png','.jpg','.jpeg','.wav','.mp3','.m4a','.mp4','.mov'}


def safe_filename(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', Path(name).name)[:140] or 'document'


def validate_magic(ext, prefix):
    if ext=='.pdf':
        return prefix.startswith(b'%PDF-')
    if ext in ('.docx','.pptx'):
        return prefix.startswith(b'PK\x03\x04')
    if ext=='.png':
        return prefix.startswith(b'\x89PNG\r\n\x1a\n')
    if ext in ('.jpg','.jpeg'):
        return prefix.startswith(b'\xff\xd8')
    if ext=='.wav':
        return prefix.startswith(b'RIFF') and prefix[8:12]==b'WAVE'
    if ext=='.mp3':
        return prefix.startswith(b'ID3') or (len(prefix)>1 and prefix[0]==255 and prefix[1]&224==224)
    if ext in ('.mp4','.mov','.m4a'):
        return prefix[4:8] == b'ftyp'
    return True


def store_upload(stream, filename, cfg):
    name=safe_filename(filename)
    ext=Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError('Unsupported media format')
    cfg.media_root.mkdir(parents=True,exist_ok=True)
    file_id=uuid.uuid4()
    path=cfg.media_root / (str(file_id)+ext)
    total=0
    digest=hashlib.sha256()
    try:
        with open(path,'xb') as out:
            while True:
                block=stream.read(1024*1024)
                if not block:
                    break
                total+=len(block)
                if total>cfg.max_upload_bytes:
                    raise ValueError('Upload exceeds size limit')
                digest.update(block)
                out.write(block)
        if total==0 or not validate_magic(ext,path.read_bytes()[:16]):
            raise ValueError('Invalid or empty media file')
        return str(file_id),path,digest.hexdigest(),total,name
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _entries_from_text(text,kind):
    if not text.strip():
        return []
    return [{'text':text[i:i+1300],'evidence':{'section':kind,'offset':i}} for i in range(0,len(text),1300)]


def _decode_utf8(path):
    return path.read_text(encoding='utf-8-sig')


def extract(path, cfg, gateway=None):
    """Return text chunks with provenance. Optional dependencies fail explicitly."""
    ext=path.suffix.lower()
    if ext in ('.txt','.md'):
        return _entries_from_text(_decode_utf8(path)[:200000],'document')
    if ext=='.pdf':
        from pypdf import PdfReader
        result=[]
        reader=PdfReader(str(path))
        for i,page in enumerate(reader.pages):
            if i >= 250 or len(result) >= 250: break
            text=(page.extract_text() or '')[:12000]
            for c in _entries_from_text(text,'page'):
                c['evidence']['page']=i+1
                result.append(c)
        return result
    if ext=='.docx':
        from docx import Document
        doc=Document(str(path))
        return _entries_from_text('\n'.join(p.text for p in doc.paragraphs if p.text.strip())[:200000],'paragraph')
    if ext=='.pptx':
        from pptx import Presentation
        prs=Presentation(str(path))
        result=[]
        for index,slide in enumerate(prs.slides):
            if index >= 250 or len(result) >= 250: break
            data='\n'.join(shape.text for shape in slide.shapes if shape.has_text_frame)[:12000]
            for c in _entries_from_text(data,'slide'):
                c['evidence']['slide']=index+1
                result.append(c)
        return result
    if ext in ('.png','.jpg','.jpeg'):
        return _extract_image(path,cfg,gateway)
    if ext in ('.wav','.mp3','.m4a','.mp4','.mov'):
        if not cfg.enable_audio_import:
            raise RuntimeError('Media transcription requires MEETING_AI_AUDIO_IMPORT=1')
        segments=_transcribe_media(path)
        if ext in ('.mp4','.mov') and cfg.enable_vision and cfg.vision_model and gateway:
            segments.extend(_video_frames(path,cfg,gateway))
        return segments
    raise ValueError('Unsupported media')


def _extract_image(path,cfg,gateway):
    if cfg.enable_vision and gateway and cfg.vision_model:
        result=gateway.chat('Describe images factually. Extract visible text without inventing content.',
                            'Describe the content and visible text of this document or slide.',
                            model=cfg.vision_model,image_bytes=path.read_bytes())
        return _entries_from_text(result,'image_vision')
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Install optional OCR dependencies or configure a vision model') from exc
    return _entries_from_text(pytesseract.image_to_string(Image.open(path)),'image_ocr')


def _transcribe_media(path):
    from faster_whisper import WhisperModel
    model_name=os.getenv('MEETING_AI_IMPORT_ASR_MODEL','small')
    device=os.getenv('MEETING_AI_IMPORT_ASR_DEVICE','cpu')
    compute=os.getenv('MEETING_AI_IMPORT_ASR_COMPUTE','int8')
    model=WhisperModel(model_name,device=device,compute_type=compute)
    segs,info=model.transcribe(str(path),vad_filter=True,beam_size=5,condition_on_previous_text=False)
    items=[]
    for seg in segs:
        if seg.text.strip():
            items.append({'text':seg.text.strip(), 'evidence':{'start_ms':int(seg.start*1000),'end_ms':int(seg.end*1000)}})
    return items


def _video_frames(path,cfg,gateway):
    """Optional sampled video-frame analysis; maximum six frames, no live video dependency."""
    results=[]
    with tempfile.TemporaryDirectory() as directory:
        pattern=str(Path(directory)/'frame_%03d.jpg')
        cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(path),
             '-vf','fps=1/60,scale=640:-2','-frames:v','6','-y',pattern]
        try:
            subprocess.run(cmd,check=True,timeout=120,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except (FileNotFoundError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
            return results
        for i,frame in enumerate(sorted(Path(directory).glob('frame_*.jpg'))):
            response=gateway.chat('Describe only visible scene facts and onscreen text. Never infer spoken audio.',
                'Describe this video frame for cross-media search.',
                model=cfg.vision_model,image_bytes=frame.read_bytes())
            results.append({'text':response[:3000],'evidence':{'frame_index':i,'sample_time_ms':i*60000,'kind':'video_frame'}})
    return results
