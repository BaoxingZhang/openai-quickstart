from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_zhipu import ChatZhipuAI
from langchain_core.runnables import RunnablePassthrough

# 创建基础规划器
planner = (
    ChatPromptTemplate.from_template("根据以下功能需求生成高级设计: {input}")
    | ChatZhipuAI(model="glm-4-flash")
    | StrOutputParser()
    | {"base_design": RunnablePassthrough()}
)

# 创建Python代码生成器
python_generator = (
    ChatPromptTemplate.from_template(
        "根据以下高级设计生成Python代码实现:\n{base_design}"
    )
    | ChatZhipuAI(model="glm-4-flash")
    | StrOutputParser()
)

# 创建Java代码生成器
java_generator = (
    ChatPromptTemplate.from_template(
        "根据以下高级设计生成Java代码实现:\n{base_design}"
    )
    | ChatZhipuAI(model="glm-4-flash")
    | StrOutputParser()
)

# 构建完整的处理链
chain = (
    planner
    | {
        "python_code": python_generator,
        "java_code": java_generator,
        "design": itemgetter("base_design"),
    }
)

# 使用示例
if __name__ == "__main__":
    requirement = "实现一个简单的计算器,支持加减乘除四则运算"
    result = chain.invoke({"input": requirement})
    
    print("高级设计:")
    print(result["design"])
    print("\nPython实现:")
    print(result["python_code"])
    print("\nJava实现:")
    print(result["java_code"])