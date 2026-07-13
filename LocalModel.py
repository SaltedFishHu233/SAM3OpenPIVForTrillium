from transformers import Sam3ImageProcessor, Sam3Model, AutoTokenizer

model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
processor = Sam3ImageProcessor.from_pretrained("facebook/sam3")
tokenizer = AutoTokenizer.from_pretrained("facebook/sam3")

model.save_pretrained("./LocalSAM3")
processor.save_pretrained("./LocalSAM3Processor")
tokenizer.save_pretrained("./LocalSAM3Tokenizer")