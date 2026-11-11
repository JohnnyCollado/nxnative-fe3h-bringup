using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using System.Collections.Generic;
using System.Linq;

using static Ryujinx.Graphics.Shader.StructuredIr.AstHelper;

namespace Ryujinx.Graphics.Shader.StructuredIr
{
    static class AstOptimizer
    {
        public static void Optimize(StructuredProgramContext context)
        {
            AstBlock mainBlock = context.CurrentFunction.MainBlock;

            // When debug mode is enabled, we disable expression propagation
            // (this makes comparison with the disassembly easier).
            if (!context.DebugMode)
            {
                AstBlockVisitor visitor = new(mainBlock);

                foreach (IAstNode node in visitor.Visit())
                {
                    if (node is AstAssignment assignment && assignment.Destination is AstOperand propVar)
                    {
                        // nxnative: a variable passed to a call stays a variable. SPIR-V passes call arguments by pointer (StructuredProgram copies each one to a temporary,
                        // GenerateCall takes the temporary's pointer); propagated, the call held the value itself, a constant or an expression, and the temporary was removed,
                        // so translation failed (KeyNotFoundException in CodeGenContext.GetLocalPointer; the shuffle helpers' constant masks: 1,850 of Fire Emblem Fortune's
                        // Weave's shaders only translated with the optimizer off)
                        if (propVar.Uses.Any(u => u is AstOperation useOp && (useOp.Inst & Instruction.Mask) == Instruction.Call))
                        {
                            continue;
                        }

                        // nxnative: an assignment whose source writes memory (an atomic, an image atomic) stays where it is, used or not. Upstream removed it when its
                        // result was unused, and with it the write: all five atomic adds of Fire Emblem Fortune's Weave's exposure histogram (6d0f6db3f0d396a3) were gone
                        // from the optimized translation, the histogram stayed empty and the picture came out dark (found by running both translations on a PC GPU:
                        // the work folder's re/csrun). A single use no longer pulls it to the use's position either (past a barrier or another atomic). As
                        // shirai91/nx2android 44eca7e671 (Tears of the Kingdom's visibility counter)
                        if (WritesMemory(assignment.Source))
                        {
                            continue;
                        }

                        bool isWorthPropagating = propVar.Uses.Count == 1 || IsWorthPropagating(assignment.Source);

                        // nxnative: a subgroup operation's result depends on which lanes are active where it executes, so its assignment is not moved to its use (which may
                        // sit in another branch or loop, with other lanes active); it is still removed when unused. Moved, the per-warp reductions of Fire Emblem
                        // Fortune's Weave's light culling (35d12d6f98fa05e2, REDUX as OpGroupNonUniformBitwiseOr) reduced over the lanes of the use's branch: wrong light
                        // masks in 31 of 120 runs on a PC GPU against the unoptimized translation (the work folder's re/csrun)
                        if (propVar.Defs.Count == 1 && isWorthPropagating && !HasSubgroupOp(assignment.Source))
                        {
                            PropagateExpression(propVar, assignment.Source);
                        }

                        if (propVar.Type == OperandType.LocalVariable && propVar.Uses.Count == 0)
                        {
                            visitor.Block.Remove(assignment);

                            context.CurrentFunction.Locals.Remove(propVar);
                        }
                    }
                }
            }

            RemoveEmptyBlocks(mainBlock);
        }

        private static bool WritesMemory(IAstNode node)
        {
            if (node is not AstOperation op)
            {
                return false;
            }

            Instruction inst = op.Inst & Instruction.Mask;

            if (inst.IsAtomic() || inst == Instruction.ImageAtomic || inst == Instruction.ImageStore || inst == Instruction.Store)
            {
                return true;
            }

            for (int i = 0; i < op.SourcesCount; i++)
            {
                if (WritesMemory(op.GetSource(i)))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool HasSubgroupOp(IAstNode node)
        {
            if (node is not AstOperation op)
            {
                return false;
            }

            switch (op.Inst & Instruction.Mask)
            {
                case Instruction.Ballot:
                case Instruction.Shuffle:
                case Instruction.ShuffleDown:
                case Instruction.ShuffleUp:
                case Instruction.ShuffleXor:
                case Instruction.VoteAll:
                case Instruction.VoteAllEqual:
                case Instruction.VoteAny:
                case Instruction.QuadBroadcast:
                case Instruction.QuadSwap:
                case Instruction.SubgroupBroadcast:
                case Instruction.SubgroupReduce:
                    return true;
            }

            for (int i = 0; i < op.SourcesCount; i++)
            {
                if (HasSubgroupOp(op.GetSource(i)))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsWorthPropagating(IAstNode source)
        {
            if (source is not AstOperation srcOp)
            {
                return false;
            }

            if (!InstructionInfo.IsUnary(srcOp.Inst))
            {
                return false;
            }

            return srcOp.GetSource(0) is AstOperand || srcOp.Inst == Instruction.Copy;
        }

        private static void PropagateExpression(AstOperand propVar, IAstNode source)
        {
            IAstNode[] uses = propVar.Uses.ToArray();

            foreach (IAstNode useNode in uses)
            {
                if (useNode is AstBlock useBlock)
                {
                    useBlock.Condition = source;
                }
                else if (useNode is AstOperation useOperation)
                {
                    for (int srcIndex = 0; srcIndex < useOperation.SourcesCount; srcIndex++)
                    {
                        if (useOperation.GetSource(srcIndex) == propVar)
                        {
                            useOperation.SetSource(srcIndex, source);
                        }
                    }
                }
                else if (useNode is AstAssignment useAssignment)
                {
                    useAssignment.Source = source;
                }
            }
        }

        private static void RemoveEmptyBlocks(AstBlock mainBlock)
        {
            Queue<AstBlock> pending = new();

            pending.Enqueue(mainBlock);

            while (pending.TryDequeue(out AstBlock block))
            {
                foreach (IAstNode node in block)
                {
                    if (node is AstBlock childBlock)
                    {
                        pending.Enqueue(childBlock);
                    }
                }

                AstBlock parent = block.Parent;

                if (parent == null)
                {
                    continue;
                }

                AstBlock nextBlock = Next(block) as AstBlock;

                bool hasElse = nextBlock != null && nextBlock.Type == AstBlockType.Else;

                bool isIf = block.Type == AstBlockType.If;

                if (block.Count == 0)
                {
                    if (isIf)
                    {
                        if (hasElse)
                        {
                            nextBlock.TurnIntoIf(InverseCond(block.Condition));
                        }

                        parent.Remove(block);
                    }
                    else if (block.Type == AstBlockType.Else)
                    {
                        parent.Remove(block);
                    }
                }
                else if (isIf && parent.Type == AstBlockType.Else && parent.Count == (hasElse ? 2 : 1))
                {
                    AstBlock parentOfParent = parent.Parent;

                    parent.Remove(block);

                    parentOfParent.AddAfter(parent, block);

                    if (hasElse)
                    {
                        parent.Remove(nextBlock);

                        parentOfParent.AddAfter(block, nextBlock);
                    }

                    parentOfParent.Remove(parent);

                    block.TurnIntoElseIf();
                }
            }
        }
    }
}
