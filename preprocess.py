from data.ImagePreprocess import ImagePreprocess


if __name__ == "__main__":
    print("Processing...")
    imageAugmentation = ImagePreprocess((256, 256), 6)
    imageAugmentation.process("pokemon", "pokemon_preprocessed")    
    print("Finished !")
