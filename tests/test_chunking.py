from scripts.ingest import split

def test_chunking_preserves_original_text_offsets():
    text='abcdefghijklmnopqrstuvwxyz'
    parts=list(split(text,10,3))
    assert all(text[offset:offset+len(fragment)]==fragment for offset,fragment in parts)
    assert parts[-1][0]+len(parts[-1][1])==len(text)
