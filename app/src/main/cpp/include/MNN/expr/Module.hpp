//
//  Module.hpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Train_Module_hpp
#define MNN_Train_Module_hpp

#include <vector>
#include <unordered_map>

#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace Express {
class MNN_PUBLIC Module {
public:
    Module()                                                                               = default;
    virtual ~Module()                                                                      = default;
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) = 0;
    Express::VARP forward(Express::VARP input);
    std::vector<Express::VARP> parameters() const;
    bool loadParameters(const std::vector<Express::VARP>& parameters);
    void setIsTraining(const bool isTraining);
    bool getIsTraining();
    void clearCache();

    const std::string& name() const {
        return mName;
    };
    void setName(std::string name) {
        mName = std::move(name);
    }
    const std::string type() const {
        return mType;
    }
    void setType(std::string type) {
        mType = std::move(type);
    }
    // Return the parameter index
    int addParameter(Express::VARP parameter);

    void setParameter(Express::VARP parameter, int index);
    static Module* createEmpty(const std::vector<Express::VARP>& parameters);
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, bool dynamic = false);
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, bool dynamic = false);

    static Module* clone(const Module* module, const bool shareParams = false);

    class CloneContext {
    public:
        CloneContext() = default;
        explicit CloneContext(const bool shareParams)
            : mShareParams(shareParams) {}
        virtual ~CloneContext() = default;

        const bool shareParams() const { return mShareParams; }

        EXPRP getOrClone(const EXPRP expr);
        VARP getOrClone(const VARP var);

    private:
        bool mShareParams = false;
        std::unordered_map<const Expr*, EXPRP> mExprMap;
        std::unordered_map<const Variable*, VARP> mVarMap;
    };

    virtual Module* clone(CloneContext* ctx) const {
        return nullptr;
    }

protected:
    void registerModel(const std::vector<std::shared_ptr<Module>>& children);
    virtual void onClearCache() {
    }

    Module* cloneBaseTo(CloneContext* ctx, Module* module) const;

private:
    void _collectParameters(std::vector<Express::VARP>& result) const;
    std::vector<std::shared_ptr<Module>> mChildren;
    std::vector<Express::VARP> mParameters;
    bool mIsTraining = true;
    std::string mName;
    std::string mType;
};

struct SubGraph {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::shared_ptr<Module> m;
};

} // namespace Train
} // namespace MNN

#endif