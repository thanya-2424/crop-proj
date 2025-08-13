"""
Create validation dataset for ML pipeline testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_validation_dataset():
    """Create a smaller validation dataset"""
    
    print("🔍 Creating validation dataset...")
    
    # Load the main dataset
    main_df = pd.read_csv('data/realistic_crop_data.csv')
    
    # Sample 1000 records for validation
    validation_df = main_df.sample(n=1000, random_state=42)
    
    # Save validation dataset
    validation_df.to_csv('data/validation_crop_data.csv', index=False)
    
    print(f"✅ Validation dataset created with {len(validation_df)} records")
    print(f"📁 Saved to: data/validation_crop_data.csv")
    
    return validation_df

if __name__ == "__main__":
    create_validation_dataset() 