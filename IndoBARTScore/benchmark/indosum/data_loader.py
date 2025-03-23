#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader for the SEACrowd/indosum dataset.
This script downloads and preprocesses the dataset for benchmarking IndoBARTScore.
"""

import os
import json
import pandas as pd
import requests
import tempfile
from datasets import load_dataset
from tqdm import tqdm

# Try importing seacrowd if available
try:
    import seacrowd as sc
    SEACROWD_AVAILABLE = True
except ImportError:
    SEACROWD_AVAILABLE = False
    print("Warning: seacrowd library not found. Will try using HuggingFace datasets directly.")


class IndoSumLoader:
    """
    Loader for the SEACrowd/indosum dataset for benchmarking IndoBARTScore.
    """
    
    def __init__(self, cache_dir=None, split='train'):
        """
        Initialize the IndoSumLoader.
        
        Args:
            cache_dir (str): Directory to cache the dataset
            split (str): Dataset split to load ('train', 'validation', or 'test')
        """
        self.cache_dir = cache_dir
        self.split = split
        self.data = None
    
    def load_dataset(self, max_samples=None, use_seacrowd=True):
        """
        Load the SEACrowd/indosum dataset.
        
        Args:
            max_samples (int): Maximum number of samples to load (None for all)
            use_seacrowd (bool): Whether to try using seacrowd library first
            
        Returns:
            list: The loaded dataset
        """
        print(f"Loading SEACrowd/indosum dataset (split: {self.split})...")
        
        # Try multiple loading strategies
        dataset = None
        errors = []
        
        # 1. Try loading with seacrowd if available and requested
        if SEACROWD_AVAILABLE and use_seacrowd:
            try:
                print("Attempting to load with seacrowd library...")
                dataset = sc.load_dataset("indosum", schema="seacrowd", split=self.split)
                print(f"Successfully loaded dataset with seacrowd ({len(dataset)} samples)")
                
                if max_samples is not None and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))
                
                return dataset
            except Exception as e:
                errors.append(f"seacrowd error: {str(e)}")
                print(f"Failed to load with seacrowd: {e}")
                dataset = None
        
        # 2. Try loading with HuggingFace datasets with trust_remote_code
        if dataset is None:
            try:
                print("Attempting to load with HuggingFace datasets (trust_remote_code=True)...")
                dataset = load_dataset("SEACrowd/indosum", split=self.split, 
                                     cache_dir=self.cache_dir, 
                                     trust_remote_code=True)
                
                if max_samples is not None and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))
                
                print(f"Successfully loaded dataset with HuggingFace datasets ({len(dataset)} samples)")
                return dataset
            
            except Exception as e:
                errors.append(f"HuggingFace datasets error: {str(e)}")
                print(f"Failed to load with HuggingFace datasets: {e}")
                dataset = None
                
        # 3. Try loading with HuggingFace datasets without trust_remote_code
        if dataset is None:
            try:
                print("Attempting to load with HuggingFace datasets (standard mode)...")
                dataset = load_dataset("SEACrowd/indosum", split=self.split, 
                                     cache_dir=self.cache_dir)
                
                if max_samples is not None and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))
                
                print(f"Successfully loaded dataset with HuggingFace datasets ({len(dataset)} samples)")
                return dataset
            
            except Exception as e:
                errors.append(f"HuggingFace datasets (standard) error: {str(e)}")
                print(f"Failed to load with standard HuggingFace datasets: {e}")
                dataset = None
        
        # If we reach here, all loading methods failed
        print("All dataset loading methods failed with the following errors:")
        for i, error in enumerate(errors):
            print(f"{i+1}. {error}")
        
        # Return None to indicate failure
        return None
    
    def preprocess(self, dataset=None, max_samples=None, use_seacrowd=True):
        """
        Preprocess the dataset for benchmarking.
        
        Args:
            dataset: Loaded dataset (if None, will load the dataset)
            max_samples (int): Maximum number of samples to process
            use_seacrowd (bool): Whether to try using seacrowd library first
            
        Returns:
            dict: Processed dataset with sources and summaries
        """
        if dataset is None:
            dataset = self.load_dataset(max_samples, use_seacrowd)
            
        # If dataset loading failed, create mock dataset
        if dataset is None:
            print("Dataset loading failed. Creating mock data instead.")
            return self.create_mock_dataset(num_samples=max_samples or 100)
        
        # Extract document and summary pairs
        sources = []
        summaries = []
        article_ids = []
        
        print("Preprocessing dataset...")
        for item in tqdm(dataset):
            try:
                # Extract the relevant fields based on the data structure
                # Different libraries might have different field names
                article_id = None
                document = None
                summary = None
                
                # Try different field names
                if 'id' in item:
                    article_id = item['id']
                
                if 'document' in item:
                    document = item['document']
                elif 'text' in item:
                    document = item['text']
                
                if 'summary' in item:
                    summary = item['summary']
                elif 'summaries' in item and isinstance(item['summaries'], list) and len(item['summaries']) > 0:
                    summary = item['summaries'][0]
                
                # If fields not found, skip this item
                if not article_id:
                    article_id = str(len(article_ids))
                
                # Skip empty entries
                if not document or not summary:
                    continue
                
                # Add to our lists
                sources.append(document)
                summaries.append(summary)
                article_ids.append(article_id)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        self.data = {
            'article_id': article_ids,
            'source': sources,
            'summary': summaries
        }
        
        print(f"Preprocessed {len(sources)} valid document-summary pairs.")
        return self.data
    
    def save_to_csv(self, output_path):
        """
        Save the preprocessed data to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
            
        Returns:
            str: Path to the saved CSV file
        """
        if self.data is None:
            raise ValueError("No data to save. Call preprocess() first.")
        
        df = pd.DataFrame(self.data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} samples to {output_path}")
        return output_path
    
    def save_to_jsonl(self, output_path):
        """
        Save the preprocessed data to a JSONL file.
        
        Args:
            output_path (str): Path to save the JSONL file
            
        Returns:
            str: Path to the saved JSONL file
        """
        if self.data is None:
            raise ValueError("No data to save. Call preprocess() first.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(len(self.data['article_id'])):
                entry = {
                    'article_id': self.data['article_id'][i],
                    'source': self.data['source'][i],
                    'summary': self.data['summary'][i]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.data['article_id'])} samples to {output_path}")
        return output_path

    def create_mock_dataset(self, num_samples=10):
        """
        Create a mock dataset for testing purposes.
        
        Args:
            num_samples (int): Number of mock samples to create
            
        Returns:
            dict: Mock dataset with sources and summaries
        """
        print(f"Creating mock dataset with {num_samples} samples...")
        
        # Sample Indonesian document-summary pairs
        mock_data = {
            'article_id': [],
            'source': [],
            'summary': []
        }
        
        # Some sample Indonesian texts and summaries
        sample_texts = [
            ("Presiden Indonesia menghadiri KTT ASEAN di Jakarta. Pertemuan tersebut membahas kerjasama ekonomi dan keamanan regional. Dalam pertemuan ini, Presiden juga menekankan pentingnya kolaborasi dalam mengatasi tantangan global seperti perubahan iklim dan pemulihan ekonomi pasca-pandemi.", 
             "Presiden Indonesia hadiri KTT ASEAN di Jakarta membahas ekonomi, keamanan, dan tantangan global."),
            
            ("Tim nasional sepak bola Indonesia berhasil mengalahkan Malaysia dengan skor 2-1 pada pertandingan persahabatan kemarin. Kedua gol Indonesia dicetak oleh striker andalan pada menit ke-20 dan menit ke-75. Pelatih menyatakan puas dengan performa tim meskipun masih ada beberapa hal yang perlu diperbaiki.", 
             "Timnas Indonesia kalahkan Malaysia 2-1 dalam laga persahabatan dengan dua gol dari striker andalan."),
            
            ("Gempa bumi berkekuatan 5,6 magnitudo mengguncang wilayah Jawa Barat pada Senin pagi. Tidak ada korban jiwa yang dilaporkan, namun beberapa bangunan mengalami kerusakan ringan. Pihak BMKG menyatakan bahwa gempa tersebut merupakan gempa tektonik akibat pergerakan lempeng.", 
             "Gempa 5,6 magnitudo guncang Jawa Barat, tidak ada korban jiwa, beberapa bangunan rusak ringan."),
            
            ("Pemerintah Indonesia meluncurkan program vaksinasi COVID-19 untuk anak-anak usia 6-11 tahun di seluruh wilayah Indonesia. Program ini menargetkan sekitar 26,5 juta anak dan akan dilaksanakan secara bertahap mulai bulan depan. Vaksin yang digunakan telah mendapatkan izin penggunaan darurat dari BPOM.", 
             "Pemerintah luncurkan vaksinasi COVID-19 untuk 26,5 juta anak 6-11 tahun mulai bulan depan."),
            
            ("Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) memperkirakan cuaca Jakarta akan cerah berawan sepanjang hari ini. Suhu diperkirakan berkisar antara 24-32 derajat Celsius dengan kelembaban sekitar 70-90%. BMKG juga memprediksi akan terjadi hujan ringan pada sore hari di beberapa wilayah Jakarta Selatan dan Jakarta Timur.", 
             "BMKG: Jakarta cerah berawan, suhu 24-32°C, kemungkinan hujan ringan sore hari di Jakarta Selatan dan Timur."),
             
            ("Menteri Pendidikan mengumumkan kebijakan baru terkait kurikulum pendidikan nasional yang akan diterapkan mulai tahun ajaran berikutnya. Kurikulum baru ini menekankan pada pengembangan keterampilan berpikir kritis, kreativitas, dan kemampuan adaptasi terhadap perkembangan teknologi. Sosialisasi akan dilakukan secara bertahap kepada para guru di seluruh Indonesia.", 
             "Menteri Pendidikan umumkan kurikulum baru fokus pada keterampilan berpikir kritis dan adaptasi teknologi."),
             
            ("Bank Indonesia mempertahankan suku bunga acuan pada level 5,75% dalam rapat dewan gubernur yang diselenggarakan hari ini. Keputusan ini diambil dengan mempertimbangkan stabilitas ekonomi nasional di tengah ketidakpastian global. Bank Indonesia juga memproyeksikan pertumbuhan ekonomi Indonesia tahun ini akan berada pada kisaran 4,7% hingga 5,5%.", 
             "BI pertahankan suku bunga acuan 5,75%, proyeksi pertumbuhan ekonomi 4,7-5,5%."),
             
            ("Festival budaya tahunan 'Pesona Indonesia' akan diselenggarakan di Yogyakarta selama seminggu penuh mulai tanggal 15 Agustus mendatang. Festival ini menampilkan berbagai seni pertunjukan tradisional, pameran kerajinan, dan kuliner khas dari seluruh nusantara. Panitia mengharapkan festival ini dapat mendorong pemulihan sektor pariwisata dan ekonomi kreatif.", 
             "Festival 'Pesona Indonesia' akan digelar di Yogyakarta 15 Agustus, tampilkan seni, kerajinan, dan kuliner nusantara."),
             
            ("Peneliti dari Institut Teknologi Bandung berhasil mengembangkan sistem deteksi dini bencana tanah longsor menggunakan teknologi sensor dan kecerdasan buatan. Sistem ini mampu memberikan peringatan hingga 24 jam sebelum terjadinya longsor dengan tingkat akurasi mencapai 85%. Inovasi ini diharapkan dapat mengurangi korban jiwa akibat bencana tanah longsor yang sering terjadi di Indonesia.", 
             "Peneliti ITB kembangkan sistem deteksi dini longsor dengan AI, berikan peringatan 24 jam sebelumnya, akurasi 85%."),
             
            ("Kementerian Perhubungan meluncurkan aplikasi transportasi publik terintegrasi untuk wilayah Jabodetabek. Aplikasi ini memungkinkan pengguna untuk merencanakan perjalanan menggunakan berbagai moda transportasi seperti MRT, LRT, Transjakarta, dan KRL dengan informasi jadwal dan rute yang akurat. Aplikasi ini juga dilengkapi dengan fitur pembayaran digital yang terintegrasi.", 
             "Kemenhub luncurkan aplikasi transportasi terintegrasi Jabodetabek untuk MRT, LRT, Transjakarta, dan KRL dengan fitur pembayaran digital.")
        ]
        
        # Generate as many samples as needed
        for i in range(num_samples):
            idx = i % len(sample_texts)
            mock_data['article_id'].append(f"mock_{i+1}")
            mock_data['source'].append(sample_texts[idx][0])
            mock_data['summary'].append(sample_texts[idx][1])
        
        self.data = mock_data
        print(f"Created mock dataset with {len(mock_data['article_id'])} samples.")
        return mock_data


def main():
    """
    Main function to demonstrate usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and preprocess the SEACrowd/indosum dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], 
                        help="Dataset split to load")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to load")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "jsonl"], 
                        help="Output file format")
    parser.add_argument("--use_mock", action="store_true", help="Use mock data instead of downloading")
    parser.add_argument("--use_seacrowd", action="store_true", help="Try loading with seacrowd library first")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize loader
    loader = IndoSumLoader(split=args.split)
    
    if args.use_mock:
        # Use mock data for testing
        data = loader.create_mock_dataset(num_samples=args.max_samples or 100)
    else:
        # Load and preprocess real data
        try:
            data = loader.preprocess(max_samples=args.max_samples, use_seacrowd=args.use_seacrowd)
        except Exception as e:
            print(f"Error preprocessing real data: {e}")
            print("Falling back to mock data...")
            data = loader.create_mock_dataset(num_samples=args.max_samples or 100)
    
    # Save to specified format
    if args.format == "csv":
        output_path = os.path.join(args.output_dir, f"indosum_{args.split}.csv")
        loader.save_to_csv(output_path)
    else:
        output_path = os.path.join(args.output_dir, f"indosum_{args.split}.jsonl")
        loader.save_to_jsonl(output_path)


if __name__ == "__main__":
    main()
