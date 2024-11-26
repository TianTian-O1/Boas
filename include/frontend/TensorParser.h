class TensorParser : public Parser {
protected:
    std::unique_ptr<ExprAST> parseTensorExpr() override;
    std::unique_ptr<ExprAST> parseTensorCreate();
    std::unique_ptr<ExprAST> parseTensorRandom();
    std::unique_ptr<ExprAST> parseMatmulExpr();
};