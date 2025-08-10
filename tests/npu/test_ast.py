import ast; code = open("test/test_gpu.txt").read(); tree = ast.parse(code); print("AST dump:"); print(ast.dump(tree, indent=2))
