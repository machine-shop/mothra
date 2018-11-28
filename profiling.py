import cProfile
import sys
import pipeline
sys.argv += ['-s', 'measurements', '-p', '-i', 'raw_images/BMNHE_500606.JPG']

pr = cProfile.Profile()

pr.enable()
pipeline.main()
pr.enable()

pr.dump_stats('antennae_results.prof')
