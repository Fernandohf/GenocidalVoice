
# GenocidalVoice

 Genocidal synthesizer.

# Lista de afazeres

1. Data Scrapping
   - [x] Pesquisar vídeos com falas do coiso
   - [x] Coletar videos de Nalbossauro falando
   - [x] Criar um dataset a partir dos vídeos
2. Filtrar apenas falas de Salnorabo com um modelo de classificação
   - [x] Separar apenas as falas de Burronaro
   - [x] Coletar falas de outras pessoas
   - [ ] Criar/treinar modelo de classificação
   - [ ] Usar o novo modelo para separar apenas as falas de Bostonaro de todos os vídeos
3. Treinar os modelos Wavenet / Waveglow

## Commands

```python processing/scrapper.py --source data/outros.csv --out data/outros/raw/```
```python processing/scrapper.py --source data/bolsoanta.csv --out data/bolsoanta/raw/```

```python processing/create_clips.py --name outros --source data/outros.csv```
```python processing/create_clips.py --name bolsoanta --source data/bolsoanta.csv```
