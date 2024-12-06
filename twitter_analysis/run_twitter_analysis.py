# run_analysis.py
import sys
from twitter_cy import analyze_network_cy

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_analysis.py <twitter_network_file>")
        sys.exit(1)
    
    analyze_network_cy(sys.argv[1])
