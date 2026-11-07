"""Ingesta piloto: conserva offsets y fuentes, indexa de forma incremental por lotes."""
import argparse, hashlib, json, os
import pyarrow.parquet as pq
from app.core import client, ensure_collection, embed, COLLECTION

def split(text, size=1000, overlap=150):
    if not 0 <= overlap < size: raise ValueError('overlap debe ser menor que size')
    pos=0
    while pos < len(text):
        end=min(pos+size,len(text)); yield pos,text[pos:end]
        if end==len(text): break
        pos=end-overlap

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/documentos_rag.parquet')
    p.add_argument('--limit',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--chunk-size',type=int,default=1000)
    p.add_argument('--overlap',type=int,default=150)
    args=p.parse_args()
    if not os.path.isfile(args.input): raise SystemExit(f'Falta Parquet: {args.input}')
    c=client();ensure_collection(c)
    parq=pq.ParquetFile(args.input)
    fields=['document_id','text','title','url','collection_name','source']
    buffer=[]; processed=0; chunks=0
    def flush():
        nonlocal chunks
        if not buffer: return
        vectors=embed([r['text'] for r in buffer])
        for record,v in zip(buffer,vectors): record['vector']=v
        c.upsert(collection_name=COLLECTION,data=buffer)
        chunks += len(buffer); buffer.clear()
    for batch in parq.iter_batches(batch_size=128,columns=fields):
        for row in batch.to_pylist():
            if args.limit and processed>=args.limit: break
            processed+=1
            text=row['text'] or ''
            for offset, chunk in split(text,args.chunk_size,args.overlap):
                raw=f"{row['document_id']}|{offset}|{args.chunk_size}|{args.overlap}"
                chunkid=hashlib.sha256(raw.encode()).hexdigest()[:32]
                buffer.append({'chunk_id':chunkid,'text':chunk,'document_id':row['document_id'],
                               'title':(row['title'] or '')[:2048], 'url':(row['url'] or '')[:4096],
                               'collection_name':row['collection_name'] or '', 'source':row['source'] or '',
                               'position':offset})
                if len(buffer)>=args.batch_size:flush()
        if args.limit and processed>=args.limit:break
    flush()
    print(json.dumps({'documents':processed,'chunks':chunks,'collection':COLLECTION},ensure_ascii=False))
if __name__=='__main__': main()
