# run_facebook_analysis.py
from facebook_cy import analyze_network_cy

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_facebook_analysis.py <facebook_network_file>")
        sys.exit(1)
    
    analyze_network_cy(sys.argv[1])
